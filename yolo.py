# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video

MIT License

Copyright (c) 2018 qqwweee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.



slighly modified to work better with DataSetLableler

"""
import os
os.environ['LD_LIBRARY_PATH']='/opt/cuda/lib64'

import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
import os, config
from keras.utils import multi_gpu_model

import cv2

class YOLO(object):
    _defaults = {
        #"model_path": 'logs/000/trained_weights_final.h5',
        #"model_path": 'model_data/trained_weights_final.h5',
        "model_path": config.modelPath,
        #"anchors_path": 'model_data/tiny_yolo_anchors.txt',
        "anchors_path": config.anchorsPath,
        "classes_path": 'classes.txt',
        "score" : 0.1,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            print('tiny')
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def preprocess_input(self, image, net_h, net_w):
        new_h, new_w, _ = image.shape
    
        # determine the new size of the image
        if (float(net_w)/new_w) < (float(net_h)/new_h):
            new_h = (new_h * net_w)/new_w
            new_w = net_w
        else:
            new_w = (new_w * net_h)/new_h
            new_h = net_h
    
        # resize the image to the new size
        resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))
    
        # embed the image into the standard letter box
        new_image = np.ones((net_h, net_w, 3)) * 0.5
        new_image[int((int(net_h)-int(new_h))//2):int((int(net_h)+int(new_h))//2), int((int(net_w)-int(new_w))//2):int((int(net_w)+int(new_w))//2), :] = resized
        new_image = np.expand_dims(new_image, 0)
    
        return new_image

    
    def detect_box_np(self, image):
        start = timer()
        height, width, channels = image.shape

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = self.preprocess_input(image, self.model_image_size[0],self.model_image_size[1])
        else:
            new_image_size = (width - (width % 32),
                              height - (height % 32))
            boxed_image = self.preprocess_input(image, new_image_size[0],new_image_size[1])
        image_data = boxed_image#np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
        #image_data /= 255.
        #image_data = np.zeros((width - (width % 32),height - (height % 32),3), np.uint8)
        #image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [height, width],
                K.learning_phase(): 0
            })
        labels = []
        for i, c in reversed(list(enumerate(out_classes))):
            labels.append({'box':(int(out_boxes[i][1]),int(out_boxes[i][0]),int(out_boxes[i][3]),int(out_boxes[i][2])),
                           'score': out_scores[i], 'label':self.class_names[c]})
        end = timer()
        print('Found {} boxes for img in {} s.'.format(len(out_boxes), end-start))
        return labels
    
    def close_session(self):
        self.sess.close()

