import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import numpy as np

class Detector():
    detection_model = None
    classes = [
        'BELLY',
        'BUTTOCKS',
        'F_BREAST',
        'F_GENITALIA',
        'M_GENETALIA',
        'M_BREAST',
    ]
    
    def __init__(self, model):
        '''
            model = Classifier('path_to_weights')
        '''
        if(isinstance(model, str)):
            Detector.detection_model = models.load_model(model, backbone_name='resnet101')
        else:
            Detector.detection_model = model
    
    def detect(self, img_path, min_prob=0.6):
        if(isinstance(img_path,str)):
            image = read_image_bgr(img_path)
        else:
            image = img_path
        image = preprocess_image(image)
        #image, scale = resize_image(image,min_side=8000, max_side=13330)
        scale = 1
        #print(scale)
        boxes, scores, labels = Detector.detection_model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        processed_boxes = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < min_prob:
                continue
            box = box.astype(int).tolist()
            label = Detector.classes[label]
            processed_boxes.append({'box': box, 'score': score, 'label': label})
            
        return processed_boxes
    
    def censor(self, img_path, out_path=None, visualize=True, parts_to_blur=['BELLY', 'BUTTOCKS', 'F_BREAST', 'F_GENITALIA', 'M_GENETALIA', 'M_BREAST'], min_prob=0.6):
        if(isinstance(img_path,str)):
            image = cv2.imread(img_path)
        else:
            image = img_path
            #image = image.astype(float)
        height, width, channels = image.shape
        boxes_Raw = Detector.detect(self, img_path, min_prob)
        boxes = [i for i in boxes_Raw if i['label'] in parts_to_blur]


        gauß = int(height/20) | 1#Filter muss ungrade sein?
        blurred  = cv2.GaussianBlur(image,(gauß, gauß), int(gauß*1.5))
        mask = np.full((height,width,3), 1, np.float)
        #blurred = np.zeros((height,width,3), np.float)
        
        pixels = 30
        blurred = cv2.resize(image, (pixels,int(pixels*height/width)),interpolation = cv2.INTER_AREA)
        blurred = cv2.resize(blurred, (width,height),interpolation = cv2.INTER_AREA)
        
        oversize = 10
        
        for box in boxes:
            TpLeft = (box['box'][0]-oversize, box['box'][1]-oversize)
            BRight = (box['box'][2]+oversize, box['box'][3]+oversize)
            mask = cv2.rectangle(mask, TpLeft, BRight, (0, 0, 0), cv2.FILLED)


        #mask = cv2.GaussianBlur(mask,(51,51), int(10))
        mask = cv2.resize(mask, (pixels,int(pixels*height/width)),interpolation = cv2.INTER_AREA)
        mask = np.round(mask+0.3)
        mask = cv2.resize(mask, (width,height),interpolation = cv2.INTER_NEAREST)
        
        mask = mask.astype(float)
        image = image.astype(float)
        blurred = blurred.astype(float)
        blurred = cv2.multiply(1-mask, blurred)

        # Multiply the background with ( 1 - alpha )
        image = cv2.multiply(mask, image)

        # Add the masked foreground and background.
        image = cv2.add(blurred, image)
        
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        for box in boxes:
            TpLeft = (box['box'][0]-oversize, box['box'][1]-oversize)
            BRight = (box['box'][2]+oversize, box['box'][3]+oversize)
            cv2.putText(image,box['label'],TpLeft, font, 1,(255,255,255),2,cv2.LINE_AA)

#        cv2.imshow("Blurred image", blurred)
#        cv2.waitKey(0)
#        cv2.imshow("Blurred image", mask)
#        cv2.waitKey(0)
        if visualize:
            cv2.imshow("Blurred image", image/255)
            cv2.waitKey(0)

        if out_path:
            cv2.imwrite(out_path, image)
        if visualize:
            cv2.destroyAllWindows()
        if not out_path and not visualize:
            return image
        return boxes_Raw

#m = Detector(detection_model)
#print(m.censor('./aN.jpg', out_path='./a.jpg', min_prob=0.4))
