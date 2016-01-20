#!/usr/bin/env python
import glob
import os
import sys
import numpy as np

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)
sys.path.append(os.path.dirname(PWD))
from settings import *

def generate_labels(output_dir='data', instance=True):
    ''' Generate labels from only the RGB images. Depth images will be ignored.
    Args:
        output_dir: directory name where train.txt, test.txt and label.txt will be generated.
        instance: whether to generate instance level or category level classification labels
    '''
    labels = []

    cwd = os.getcwd()
    os.chdir(rgbd_dataset)
    img_files = glob.glob('*/*/*_crop.png')
    os.chdir(cwd)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    np.random.seed(123)
    np.random.shuffle(img_files)

    with open(output_dir+'/train.txt','w') as f:
        for img_file in img_files:
            if instance:
                label = img_file.split('/')[1] # For instance level classification
            else:
                label = img_file.split('/')[0] # For category level classification   
            if label not in labels:
                    labels.append(label)
            f.write('%s %s\n'%(img_file, labels.index(label)))
    
    with open(output_dir+'/labels.txt','w') as f:
        for i, label in enumerate(labels):
            f.write('%s %s\n'%(i, label))

    os.chdir(rgbd_dataset_eval)
    img_files = glob.glob('*/*/*_crop.png')
    os.chdir(cwd)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    np.random.shuffle(img_files)
    with open(output_dir+'/test.txt','w') as f:
        for img_file in img_files:
            if instance:
                label = img_file.split('/')[1] # For instance level classification
            else:
                label = img_file.split('/')[0] # For category level classification   
            f.write('%s %s\n'%(img_file, labels.index(label)))
    
    print "Labels written! To create the lmdb file run the following commands:"
    print "PATH_TO_CAFFE/build/tools/convert_imageset %s %s/train.txt PATH_TO_TRAIN_LMBD -backend lmdb -resize_height 227 -resize_width 227" % (rgbd_dataset, output_dir)
    print "PATH_TO_CAFFE/build/tools/convert_imageset %s %s/test.txt PATH_TO_TEST_LMBD -backend lmdb -resize_height 227 -resize_width 227" % (rgbd_dataset_eval, output_dir) 

if __name__ == '__main__':
    generate_labels()
