#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import json
import logging
import requests
import functools

import rtmidi
from rtmidi.midiutil import open_midiport

from io import BytesIO
from os.path import dirname, join, isfile, abspath
from collections import deque

from tornado.options import define, options, parse_command_line
from tornado import ioloop

logger = logging.getLogger(__file__)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("connectionpool").setLevel(logging.CRITICAL)

PROJECT_ROOT = abspath(join(dirname(__file__), '..'))


class MidiHandler(object):

    def __init__(self, midi_in, in_port, midi_out, out_port, endpoint='127.0.0.1:8080/mode'):
        self.midi_in = midi_in
        self.in_port = in_port
        self.midi_out = midi_out
        self.out_port = out_port
        self.states = [False for x in xrange(16)]
        self.endpoint = endpoint
        self._wallclock = time.time()

    def __call__(self, event, data=None):
        message, deltatime = event
        self._wallclock += deltatime 
        msg_type, key, value = message
        if msg_type == 144:
            if value > 0:
                logger.info('Note ON')
                logger.info('{}: Note {}@{}'.format(self.in_port, key, value))
            elif value == 0:
                logger.info('Note OFF')
        elif msg_type == 176 and key <= 16:
            if value == 0:
                logger.info('Control {} OFF'.format(key))
                self.send_message(key)
            elif value == 127:
                logger.info('Control {} ON'.format(key))
                self.send_message(key)
            else:
                logger.warn(value)
        elif msg_type == 176 and key > 16:
            logger.info('Control {}@{}'.format(key, value))
        elif msg_type == 160:
            logger.debug('Aftertouch {}@{}'.format(key, value))
        else:
            logger.error('WTF?')
            logger.error(message)

    def send_message(self, filter_id):
        payload = json.dumps({'filter_id': filter_id})
        r = requests.post(self.endpoint, data=payload)
        response = r.json()
        if response.has_key('error'):
            logger.error(response['error'])
        elif response.has_key('filter_id'):
            filter_id = response['filter_id']
            new_state = response['new_state']
            self.states[filter_id] = new_state
            logger.info('Filter: {} ==> {}'.format(filter_id, new_state))
        elif response.has_key('reset'):
            self.states = [False for x in xrange(16)]
            logger.warn('Reset: {}'.format(response['reset']))
        else:
            logger.warn(response)


def close_port(ch):
    ch.close_port()
    del ch


def main():
    define('host', default='127.0.0.1:8080', help='whity server host', type=str)
    parse_command_line()
    
    # get MIDI input
    try:
        midi_in, midi_in_port = open_midiport(1, 'input')
    except (EOFError, KeyboardInterrupt):
        logger.warn('Bye!')
        sys.exit()
    
    # get MIDI output
    try:
        midi_out, midi_out_port = open_midiport(1, 'output')
    except (EOFError, KeyboardInterrupt):
        logger.warn('Bye!')
        sys.exit()

    if not midi_in_port or not midi_out_port:
        logger.error('No MIDI in/out ports available. Please connect a device!')
        sys.exit()

    try:
        logger.info('Listening for incoming/outgoing MIDI notes ...')
        api_endpoint = '{}/mode'.format(options.host)
        logger.debug('API Endpoint: {}'.format(api_endpoint))
        midi_handler = MidiHandler(midi_in, midi_in_port,
                                   midi_out, midi_out_port,
                                   endpoint=api_endpoint)
        midi_in.set_callback(midi_handler)
        logger.warn('Press Control-C to exit.')
        event_loop = ioloop.IOLoop.instance()
        event_loop.start()
    except KeyboardInterrupt:
        logger.warn('Closing MIDI ports ...')
        close_port(midi_out)
        close_port(midi_in)
    except Exception as e:
        logger.error(e)
    finally:
        sys.exit('Done')


if __name__ == '__main__':
    main()

