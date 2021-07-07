#import tensorflow as tf
import sys
import os
import json
#import dataset_util
import PIL
import cv2

from os import listdir
from os.path import isfile, join
import os

import xmltodict

#flags = tf.app.flags
#flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#FLAGS = flags.FLAGS

#writer_train = tf.python_io.TFRecordWriter("/media/wassimea/Storage/invest_ottawa/car_detection/train.record")   #output file
#writer_test = tf.python_io.TFRecordWriter("/media/wassimea/Storage/invest_ottawa/car_detection/test.record")   #output file

images_folder = "/home/wassimea/Desktop/Images/Original/"
annotations_folder = "/home/wassimea/Desktop/Annotations/Anno_XML/"


def create_tf_example(filename, mode):
  # TODO(user): Populate the following variables from your example.

  


  #filename = "/Data2TB/SMATS/augmented/8bit3c/train/" + example # Filename of the image. Empty if image is not from file
  #encoded_image_data = None # Encoded image bytes


  #with open(filename) as f:
  #  content = f.readlines()
  #content = [x.strip() for x in content]
  #new_img = PIL.Image.new("L", (480, 640))
  #new_img.putdata(content)

  #with tf.gfile.GFile(filename, 'rb') as fid:
  #  encoded_jpg = fid.read()

  height = 960
  width = 1280


  with open(annotations_folder + filename.replace(".JPG", "_LMformat.xml")) as xml_file:
    data_dict = xmltodict.parse(xml_file.read())
    
  y = 1


  #for i in range(0,len(jsondata)):
  for frame in jsondata:
    filename = frame
    if os.path.exists(parent_folder + filename):
      xmins = [] 
      xmaxs = [] 
      ymins = [] 
      ymaxs = []
      classes_text = [] # List of string class name of bounding box (1 per box)
      classes = [] # List of integer class id of bounding box (1 per box)

      with tf.gfile.GFile(parent_folder + filename, 'rb') as fid:
        encoded_jpg = fid.read()
        #if(jsondata['frames'][i]['file'] == example): 
      for j in range(0,len(jsondata[frame]['annotations'])):
          if(jsondata[frame]['annotations'][j]['category'] == 'Head'):   #we are only concerned with heads
            xmin = (jsondata[frame]['annotations'][j]['x'])/640
            xmax = (jsondata[frame]['annotations'][j]['x'] + jsondata[frame]['annotations'][j]['width'])/640
            ymin = (jsondata[frame]['annotations'][j]['y'])/480
            ymax = (jsondata[frame]['annotations'][j]['y'] + jsondata[frame]['annotations'][j]['height'])/480
            #im = cv2.imread(parent_folder + filename)
            #cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            #cv2.imshow("im",im)
            #cv2.waitKey()
            if xmin < 0:
              xmin = 0
            if ymin < 0:
              ymin = 0
            if xmax > 1:
              xmax = 1
            if ymax > 1:
              ymax = 1
            xmins.append(xmin)  
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            classes_text.append('head')
            classes.append(1)
      tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/filename': dataset_util.bytes_feature(str.encode(filename)),
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/encoded': dataset_util.bytes_feature(encoded_jpg),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
      }))
      if os.path.exists(test_folder + filename):
        writer_test.write(tf_example.SerializeToString())
      else:
        writer_train.write(tf_example.SerializeToString())
  writer_train.close()
  writer_test.close()

def main(): 
  image_filenames = [f for f in listdir(images_folder) if isfile(join(images_folder, f))]
  
  for image_filename in image_filenames:
    create_tf_example(image_filename, "train")
  print('Argument List:', str(sys.argv))
  #writer = tf.python_io.TFRecordWriter("/Data2TB/SMATS/augmented/records/8bit3c/train.record")   #output file
  create_tf_example()
  #examples = os.listdir("/Data2TB/SMATS/augmented/8bit3c/train")

  #for example in examples:
  #  print(str(count) + ":" + example)
  #  tf_example = create_tf_example(example)
  #  writer.write(tf_example.SerializeToString())
  #  count = count + 1

  #writer.close()


if __name__ == '__main__':
  main()