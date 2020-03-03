#!/bin/python 

import numpy as np
import pandas as pd
import os
from sklearn.svm.classes import SVC
import cPickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print "Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0])
        print "model_file -- path of the trained svm file"
        print "feat_dir -- dir of feature files"
        print "feat_dim -- dim of features; provided just for debugging"
        print "output_file -- path to save the prediction score"
        exit(1)

    # svm code here =====
    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    
    video_list_file = 'list/val.video'

    svm = cPickle.load(open(model_file,"rb"))
    video_list = []
    fopen = open(video_list_file, 'r')

    for line in fopen.readlines():
        video_list.append(line.replace('\n',''))
    fopen.close()

    fopen = open(output_file, 'w') 
    for video in video_list:
        feat_vec = pd.read_csv(feat_dir + video, sep = ';', header = None, dtype = 'float64')
        pred = svm.decision_function(feat_vec)
        fopen.write(str(pred[0]) + '\n')
    fopen.close()

    print 'SVM tested successfully!'


