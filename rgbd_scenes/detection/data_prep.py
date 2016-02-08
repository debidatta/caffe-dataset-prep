#!/usr/bin/env python
import glob
import os
import sys
import numpy as np
import scipy.io

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)
sys.path.append(os.path.dirname(PWD))
from settings import *

labels = []

def get_label_and_bbox(img_file, instance):
    annotation_file = img_file.split('/')[1]
    mat = scipy.io.loadmat(os.path.join(rgbd_scenes_dataset, img_file.split('/')[0], annotation_file+'.mat'))
    frame_number = int(img_file.split('_')[-1][:-4])-1
    frame_data = mat['bboxes'][0][frame_number][0]
    label_list = []
 
    if len(frame_data) == 0:
        return label_list
 
    for obj in frame_data:
        if instance:
            label = str(obj[0][0]) + '_' + str(obj[1][0][0]) # For instance level classification
        else:
            label = str(obj[0][0]) # For category level classification   
        if label not in labels:
                    labels.append(label)
        x1, y1, x2, y2 = obj[4][0][0],  obj[2][0][0],  obj[5][0][0],  obj[3][0][0]
        label_list.append([label, [x1, y1, x2, y2]])
    return label_list

def generate_labels(output_dir='data', instance=False, train_test_split=0.8):
    ''' Generate labels and detection boxes from only the RGB images. Depth images and background
        scenes will be ignored.
    Args:
        output_dir: directory name where train.txt, test.txt and label.txt will be generated.
        instance: whether to generate instance level or category level classification labels
    '''
    cwd = os.getcwd()
    os.chdir(rgbd_scenes_dataset)
    img_files = glob.glob('*/*/[!background]*[!_depth].png')
    os.chdir(cwd)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'Annotations')):
        os.mkdir(os.path.join(output_dir, 'Annotations'))
    if not os.path.exists(os.path.join(output_dir, 'ImageSets')):
        os.mkdir(os.path.join(output_dir, 'ImageSets'))
    
    np.random.seed(123)
    np.random.shuffle(img_files)

    with open(output_dir+'/ImageSets/train.txt','w') as f:
        for idx in xrange(0, int(train_test_split * len(img_files))):
            img_file = img_files[idx]
            label_list = get_label_and_bbox(img_file, instance)
            with open(output_dir+'/Annotations/'+str(idx)+'.txt','w' ) as anno_f:
                for label, bbox in label_list:
                    anno_f.write('%s %s %s %s %s\n'%(bbox[0], bbox[1], bbox[2], bbox[3], labels.index(label)))
            f.write('%s %s\n'%(idx, img_file))
 
    with open(output_dir+'/ImageSets/test.txt','w') as f:
        for idx in xrange(int(train_test_split * len(img_files)), len(img_files)):
            img_file = img_files[idx]
            label_list = get_label_and_bbox(img_file, instance)
            with open(output_dir+'/Annotations/'+str(idx)+'.txt','w' ) as anno_f:
                for label, bbox in label_list:
                    anno_f.write('%s %s %s %s %s\n'%(bbox[0], bbox[1], bbox[2], bbox[3], labels.index(label)))
            f.write('%s %s\n'%(idx, img_file))
    
    with open(output_dir+'/labels.txt','w') as f:
        for i, label in enumerate(labels):
            f.write('%s %s\n'%(i, label))

    print "Labels written!"

if __name__ == '__main__':
    generate_labels(train_test_split=1)
