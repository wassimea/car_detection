import os
import io
import glob
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import random

from PIL import Image
from object_detection.utils import dataset_util

from os import listdir
from os.path import isfile, join
import os
import xmltodict
import cv2

'''
this script automatically divides dataset into training and evaluation (10% for evaluation)
this scripts also shuffles the dataset before converting it into tfrecords
if u have different structure of dataset (rather than pascal VOC ) u need to change
the paths and names input directories(images and annotation) and output tfrecords names.
(note: this script can be enhanced to use flags instead of changing parameters on code).

default expected directories tree:
dataset- 
   -JPEGImages
   -Annotations
    dataset_to_tfrecord.py   


to run this script:
$ python dataset_to_tfrecord.py 

'''

images_folder = "/home/wassimea/Desktop/cv/car_detection/VOCtrainval_25-May-2011/TrainVal/VOCdevkit/VOC2011/JPEGImages/"
annotations_folder = "/home/wassimea/Desktop/cv/car_detection/VOCtrainval_25-May-2011/TrainVal/VOCdevkit/VOC2011/Annotations/"

writer_train = tf.python_io.TFRecordWriter('train.record')     
writer_test = tf.python_io.TFRecordWriter('test.record')

def create_example(image_filename):
        #image_filename = "2011_006534.jpg"
        image_filename = "2009_005308.jpg"
        #process the xml file
        with open(annotations_folder + image_filename.replace(".jpg", ".xml")) as xml_file:
            annotations = xmltodict.parse(xml_file.read())["annotation"]

        img = cv2.imread(images_folder + image_filename)
        height, width, channels = img.shape

        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []

        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        if isinstance(annotations['object'], list):
            for i in range(len(annotations['object'])):
                if annotations["object"][i]["name"] == "car":
                    xmins.append(float(annotations["object"][i]["bndbox"]["xmin"]) / width)
                    ymins.append(float(annotations["object"][i]["bndbox"]["ymin"]) / height)
                    xmaxs.append(float(annotations["object"][i]["bndbox"]["xmax"]) / width)
                    ymaxs.append(float(annotations["object"][i]["bndbox"]["ymax"]) / height)
                    classes.append(1)
        else:
            if annotations["object"]["name"] == "car":
                xmins.append(float(annotations["object"]["bndbox"]["xmin"]) / width)
                ymins.append(float(annotations["object"]["bndbox"]["ymin"]) / height)
                xmaxs.append(float(annotations["object"]["bndbox"]["xmax"]) / width)
                ymaxs.append(float(annotations["object"]["bndbox"]["ymax"]) / height)
                classes.append(1)


        with tf.gfile.GFile(images_folder + image_filename, 'rb') as fid:
            encoded_jpg = fid.read()

		
        #create TFRecord Example
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(image_filename.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/label': dataset_util.int64_list_feature(classes)
        }))	
        return example	
		
def main(_):
    image_filenames = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
    
    random.shuffle(image_filenames)   #shuffle files list
    i=1 
    tst=0   #to count number of images for evaluation 
    trn=0   #to count number of images for training
    for image_filename in image_filenames:
      example = create_example(image_filename)
      if (i%10)==0:  #each 10th file (xml and image) write it for evaluation
         writer_test.write(example.SerializeToString())
         tst=tst+1
      else:          #the rest for training
         writer_train.write(example.SerializeToString())
         trn=trn+1
      i=i+1
      print(image_filename)
    writer_test.close()
    writer_train.close()
    print('Successfully converted dataset to TFRecord.')
    print('training dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)	
	
if __name__ == '__main__':
    tf.app.run()
