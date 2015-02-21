#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
from io import BytesIO
from os.path import dirname, join, isfile
from collections import deque

import tornado.ioloop

from jinja2 import Environment, FileSystemLoader
from jinja2.exceptions import TemplateNotFound

from tornado.web import (
    Application,
    RequestHandler,
    stream_request_body,
    HTTPError
)
from tornado.options import define, options, parse_command_line


images = deque(maxlen=3)


class Base(RequestHandler):
    @property
    def env(self):
        return self.application.env

    def render(self, template, **kwargs):
        try:
            template = self.env.get_template(template)
        except TemplateNotFound:
            raise HTTPError(404)
        self.write(template.render(kwargs))


class MainHandler(Base):
    def get(self):
        self.render('index.html')


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



class ImageHandler(RequestHandler):

    last_image = None

    def process_image(self, image):
        with open('image.jpg', 'wb') as f:
            f.write(image.getvalue())
        img = cv2.imread('image.jpg')
        #img = detect_face(img)
        #detect_person(img)
        #result, img = cv2.imencode('*.jpeg', img)
        #print(type(img))
        #print(dir(img))
        #return img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)

        #kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        #close = cv2.morphologyEx(img, cv2.MORPH_CLOSE,kernel1)
        #div = np.float32(img)/(close)
        #img = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
        #res = np.hstack((img, equ))
        #img = cv2.GaussianBlur(img, (9, 9), 2.0)
        r = 60
        img = cv2.Canny(img, r, 3 * r)

        cv2.imwrite('image.jpg', img)
        return open('image.jpg', 'rb').read()
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

    def get(self):
        try:
            image = images.pop()
        except IndexError:
            image = ImageHandler.last_image
        if not image:
            self.write('No images received')
            return
        else:
            ImageHandler.last_image = image
        image = self.process_image(image)
        #image = image.getvalue()
        self.set_header('Content-type', 'image/jpg')
        self.set_header('Content-length', len(image))
        self.write(image)


@stream_request_body
class UploadHandler(RequestHandler):

    def post(self):
        self.f.flush()
        images.append(self.f)
        self.finish()

    def prepare(self):
        self.f = BytesIO()

    def data_received(self, data):
        self.f.write(data)


class WhityApp(Application):
    def __init__(self, root, debug_mode=False):
        debug = debug_mode or isfile(join(root, 'debug'))
        handlers = [
            (r'/upload/?', UploadHandler),
            (r'/image/?', ImageHandler),
            (r'/', MainHandler)
        ]
        settings = {
            'debug': debug
        }
        template_path = join(root, 'templates')
        self.env = Environment(loader=FileSystemLoader(template_path))
        super(WhityApp, self).__init__(handlers, **settings)


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
