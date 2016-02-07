#!/usr/bin/env python
import glob
import os
import sys
import numpy as np

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)
sys.path.append(os.path.dirname(PWD))
from settings import *

labels = []

def generate_labels(output_dir='data', train_val_split=0.8, exclude_labels=[]):
    ''' Generate labels from only the Raw RGB images. Depth images will be ignored.
    Args:
        output_dir: directory name where train.txt, test.txt and label.txt will be generated.
        train_val_split: Percent split for training and validation datasets 
        exclude_labels: Exclude the following labels while training
    '''
    with open(objects_path,'r') as f:
        labels = [x.strip() for x in f.readlines() if x.strip not in exclude_labels]   

    cwd = os.getcwd()
    os.chdir(bigbird_dataset)
    img_files = glob.glob('*/*.jpg')
    os.chdir(cwd)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    np.random.seed(123)
    np.random.shuffle(img_files)
    split_idx = int(len(img_files)*train_val_split)
    train_imgs = img_files[0:split_idx]
    test_imgs = img_files[split_idx:]

    with open(output_dir+'/train.txt','w') as f:
        for img_file in train_imgs:
            label = img_file.split('/')[0] # For category level classification
            f.write('%s %s\n'%(img_file, labels.index(label)))
    
    with open(output_dir+'/labels.txt','w') as f:
        for i, label in enumerate(labels):
            f.write('%s %s\n'%(i, label))

    with open(output_dir+'/test.txt','w') as f:
        for img_file in test_imgs:
            label = img_file.split('/')[0] # For category level classification   
            f.write('%s %s\n'%(img_file, labels.index(label)))
    
    print "Labels written! To create the lmdb file run the following commands:"
    print "PATH_TO_CAFFE/build/tools/convert_imageset %s %s/train.txt PATH_TO_TRAIN_LMBD -backend lmdb -resize_height 227 -resize_width 227" % (bigbird_dataset, output_dir)
    print "PATH_TO_CAFFE/build/tools/convert_imageset %s %s/test.txt PATH_TO_TEST_LMBD -backend lmdb -resize_height 227 -resize_width 227" % (bigbird_dataset, output_dir) 

if __name__ == '__main__':
    generate_labels()
