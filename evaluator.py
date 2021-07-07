import tensorflow as tf
import numpy as np
import cv2
import json
import PIL
import matplotlib
from PIL import Image
from os import listdir
from os.path import isfile, join
import time
import os

PATH_TO_MODEL = "/media/wassimea/Storage/invest_ottawa/car_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb"

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class CarDetector(object):

    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        config = tf.ConfigProto(device_count = {'GPU': 1})
        self.sess = tf.Session(config=config,graph=self.detection_graph)

    def get_classification(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        x = len(boxes)
        arrb = boxes
        return arrb, scores, classes, num


def evaluate():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920,1080))
    images_path = "/media/wassimea/Storage/invest_ottawa/kanata_images/"
    image_filenames = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    pingit = CarDetector()

    index = 999
    while os.path.exists(images_path + str(index) + ".png"):
        img = cv2.imread(images_path + str(index) + ".png")
        height, width, channels = img.shape
        disp_image = img.copy()
        img = img[0:height - 400, 0:width]
        img = img[...,::-1]



        result = pingit.get_classification(img)
        boxes, scores, classes, num = result

        for i in range(0,len(scores[0])):
            if(scores[0][i] > 0.05)and int(classes[0][i]) in [2,3,4,6,8,9]:
                v = boxes[0][i]
                x1 = int(boxes[0][i][1] * width)
                y1 = int(boxes[0][i][0] * (height - 400))
                x2 = int(boxes[0][i][3] * width)
                y2 = int(boxes[0][i][2] * (height - 400))
                cv2.rectangle(disp_image,(x1,y1), (x2,y2),(0,0,255), 3)
        
        #cv2.imshow("im", disp_image)
        #cv2.waitKey(1)
        index += 1
        out.write(disp_image)


def main(_):
    evaluate()
if __name__ == '__main__':
  tf.app.run()