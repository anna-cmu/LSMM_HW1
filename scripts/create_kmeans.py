#!/bin/python
import numpy as np
import pandas as pd
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0])
        print "kmeans_model -- path to the kmeans model"
        print "cluster_num -- number of cluster"
        print "file_list -- the list of videos"
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = cPickle.load(open(kmeans_model,"rb"))
    
    # kmeans code here =====        
    fread = open(file_list, "r")
    for line in fread.readlines():
        file_name = line.replace('\n','')
        mfcc_path = "mfcc/" + file_name + ".mfcc.csv"
        fw = open('kmeans/' + file_name,'w')
        cluster_histogram = np.zeros(cluster_num)

        if os.path.exists(mfcc_path) == True:
            array = pd.read_csv(mfcc_path, sep = ";", header = None)
            pred = kmeans.predict(array)

            for x in pred:
                cluster_histogram[x] += 1
            for m in xrange(cluster_num):
                cluster_histogram[m] /= len(pred)
        else:
            cluster_histogram.fill(1.0/cluster_num)  

        line = str(cluster_histogram[0])
        for m in range(1, cluster_num):
            line += ';' + str(cluster_histogram[m])
        fw.write(line + '\n')
        fw.close()

    print "K-means features generated successfully!"
