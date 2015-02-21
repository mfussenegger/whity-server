#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import json
import os
from collections import OrderedDict
from io import BytesIO
from os.path import dirname, join, isfile

import tornado.ioloop

from jinja2 import Environment, FileSystemLoader
from jinja2.exceptions import TemplateNotFound

from tornado.websocket import WebSocketHandler
from tornado.web import (
    Application,
    RequestHandler,
    stream_request_body,
    HTTPError
)
from tornado.options import define, options, parse_command_line


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def detect(img, cascade):
    rects = cascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


cascade_fn = '/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml'
nested_fn = '/usr/share/opencv/haarcascades/haarcascade_eye.xml'
cascade = cv2.CascadeClassifier(cascade_fn)
nested = cv2.CascadeClassifier(nested_fn)


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray, cascade)
    vis = img.copy()
    draw_rects(vis, rects, (0, 255, 0))
    for x1, y1, x2, y2 in rects:
        roi = gray[y1:y2, x1:x2]
        vis_roi = vis[y1:y2, x1:x2]
        subrects = detect(roi.copy(), nested)
        draw_rects(vis_roi, subrects, (255, 0, 0))
    return vis


hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )


def detect_person(img):
    found, w = hog.detectMultiScale(
        img, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    draw_detections(img, found)
    draw_detections(img, found_filtered, 3)



#img = detect_face(img)
#detect_person(img)
#result, img = cv2.imencode('*.jpeg', img)
#print(type(img))
#print(dir(img))
#return img
#kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#close = cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel1)
#div = np.float32(img)/(close)
#img = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))

#print('fromstring')
##data = np.asarray(bytearray(image.getvalue()), dtype=np.uint8)
#image.seek(0)
#data = np.fromstring(image.getvalue(), dtype=np.uint8)
#print('decode')
#img = cv2.imdecode(data, -1)
#if img:
#    print('cvtColor')
#    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#return image

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def equalize_hist(image):
    return cv2.equalizeHist(grayscale(image))


def canny(image):
    r = 60
    return cv2.Canny(image, r, 3 * r)


def blur(image):
    return cv2.GaussianBlur(image, (9, 9), 2.0)


def good_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(image, (x,y), 3, 255, -1)

    return image


def invert(image):
    return (255 - image)


def threshold_adaptive_gaussian(img):
    if len(img.shape) == 3:
        img = cv2.split(img)[0]
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


def threshold_adaptive_mean(img):
    if len(img.shape) == 3:
        img = cv2.split(img)[0]
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    return img


filters = {
    1: detect_face,
    2: grayscale,
    3: equalize_hist,
    4: canny,
    5: blur,
    6: good_features,
    7: invert,
    8: threshold_adaptive_mean,
    9: threshold_adaptive_gaussian
}

active_filters = OrderedDict()


class ModeHandler(RequestHandler):
    def post(self):
        payload = json.loads(self.request.body)
        try:
            new_filter = int(payload.get('filter_id'))
        except ValueError:
            return self.write({"error": "invalid filter format"})
        if new_filter == 16:
            active_filters.clear()
            return self.write({'reset': 'ok'})
        if new_filter in active_filters:
            del active_filters[new_filter]
            self.write({
                "new_state": False,
                "filter_id": new_filter
            })
        else:
            if new_filter not in filters:
                self.write({"error": "unknown filter"})
            else:
                active_filters[new_filter] = filters[new_filter]
                self.write({
                    "new_state": True,
                    "filter_id": new_filter
                })


def process_image(image):
    with open('image.jpg', 'wb') as f:
        f.write(image.getvalue())
    img = cv2.imread('image.jpg')

    for f in active_filters.values():
        try:
            img = f(img)
        except Exception as e:
            print(e)
            continue

    cv2.imwrite('image.jpg', img)
    return open('image.jpg', 'rb').read()


@stream_request_body
class UploadHandler(RequestHandler):

    def post(self):
        self.f.flush()
        image = process_image(self.f)
        WebSocket.send_image(image)
        self.finish()

    def prepare(self):
        self.f = BytesIO()

    def data_received(self, data):
        self.f.write(data)


class WebSocket(WebSocketHandler):

    clients = set()

    def open(self):
        WebSocket.clients.add(self)

    def on_message(self, message):
        pass

    def on_close(self):
        WebSocket.clients.remove(self)

    @classmethod
    def send_image(cls, image):
        for client in cls.clients:
            client.write_message(image, binary=True)


class MainHandler(RequestHandler):
    def render(self, template, **kwargs):
        try:
            template = self.application.env.get_template(template)
        except TemplateNotFound:
            raise HTTPError(404)
        self.write(template.render(kwargs))

    def get(self):
        self.render('index.html')


class WhityApp(Application):
    def __init__(self, root, debug_mode=False):
        debug = debug_mode or isfile(join(root, 'debug'))
        handlers = [
            (r'/upload/?', UploadHandler),
            (r'/websocket/?', WebSocket),
            (r'/mode/?', ModeHandler),
            (r'/', MainHandler)
        ]
        settings = {
            'debug': debug
        }
        template_path = join(root, 'templates')
        self.env = Environment(loader=FileSystemLoader(template_path))
        super(WhityApp, self).__init__(handlers, **settings)


def _convert(image_path):
    img = cv2.imread(image_path)
    img = cv2.GaussianBlur(img, (5, 5), 10)

    img = cv2.resize(img, (0,0), fx=0.2, fy=0.2) 

    triangles = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                i = cv2.Canny(gray, 40, 120)
                i = cv2.dilate(i, None)
            else:
                _, i = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
        countours, hierarchy = cv2.findContours(
            i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in countours:
            cnt_len = cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
            if len(cnt) == 3:
                print(cnt)
                cnt = cnt.reshape(-1, 2)
                triangles.append(cnt)


    #img = blur(img)
    #mg = equalize_hist(img)
    #img = invert(img)
    r = 40
    #img = cv2.Canny(img, r, 3 * r)
    #img = invert(img)
    #img = invert(img)
    #img = good_features(img)

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #corners = cv2.goodFeaturesToTrack(img, 3, 0.99, 10)
    #corners = np.int0(corners)
    #for i in corners:
    #    x,y = i.ravel()
    #    cv2.circle(img, (x,y), 3, 255, -1)
    cv2.drawContours(img, triangles, -1, (0, 255, 0), 3)
    cv2.imwrite('tmp.jpg', img)


def convert():
    from argparse import ArgumentParser
    p = ArgumentParser('convert')
    p.add_argument('path')
    path = p.parse_args().path
    IMAGE_VIEWER = ('/usr/bin/open'
                    if os.path.exists('/usr/bin/open')
                    else '/usr/bin/xdg-open')
    from subprocess import call
    _convert(path)
    call([IMAGE_VIEWER, 'tmp.jpg'])


def main():
    define('port', default=8080, help='run on the given port', type=int)
    parse_command_line()

    here = dirname(__file__)
    project_root = join(here, '..')

    app = WhityApp(project_root)
    app.listen(options.port)
    try:
        print('Starting on http://localhost:{0}/'.format(options.port))
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        sys.exit('Bye')


if __name__ == '__main__':
    main()
