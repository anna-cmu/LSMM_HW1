#!/bin/python 

import numpy as np
import pandas as pd
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0])
        print "event_name -- name of the event (P001, P002 or P003 in Homework 1)"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features"
        print "output_file -- path to save the svm model"
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    
    # svm code here =====
    video_list_file = 'list/train'
    video_list = []; label_string = ''
    fopen = open(video_list_file, 'r')
    
    for line in fopen.readlines():
        splits = line.replace('\n','').split(' ')
        video_list.append(splits[0])
        if splits[1] == event_name:
            label_string += '1 '
        else:
            label_string += '0 '
    fopen.close()
    
    label_vec = np.fromstring(label_string.strip(), dtype=int, sep=' ')
    
    video_num = len(video_list)
    feat_mat = np.zeros([video_num, feat_dim])
    
    for i in xrange(video_num):
        feat_vec = pd.read_csv(feat_dir + video_list[i], sep = ';', header = None, dtype = 'float64')
        feat_mat[i,:] = feat_vec
    
    svm = SVC(kernel = 'rbf', gamma=0.9)
    svm.fit(feat_mat, label_vec) 

    cPickle.dump(svm, open(output_file,"wb"))

    print 'SVM trained successfully for event %s!' % (event_name)
