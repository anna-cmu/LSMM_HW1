#!/bin/python
import numpy as np
import os
import cPickle
from sklearn.cluster.k_means_ import KMeans
import sys
import glob

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Usage: {0} vocab_file, file_list".format(sys.argv[0])
        print "file_list -- the list of videos"
        exit(1)
    
    # asrfeat code here =====
    file_list = sys.argv[1]

    paths = '../asrs/*.txt'
    words_list = []

    for path in glob.glob(paths):
        f = open(path, 'r')    
        for line in f.readlines():
            words = line.replace('\n','').replace('.', '').replace(',', '').split(' ')        
            words_list.extend(words)

    words_arr = np.array(words_list)
    (vocab, counts) = np.unique(words_arr, return_counts=True)
    vocab = vocab[counts.argsort()[::-1][0:1100]].tolist()

    vocab_size = len(vocab)
    
    fread = open(file_list, "r")

    for line in fread.readlines():
        asr_path = "../asrs/" + line.replace('\n','') + ".txt"
        fw = open('asrfeat/' + line.replace('\n',''),'w')

        cluster_histogram = np.zeros(vocab_size)
        total_occur = 0

        if os.path.exists(asr_path) == True:
            fread = open(asr_path, 'r')
            words_list = []
            for lines in fread.readlines():
                words = lines.replace('\n','').replace('.', '').replace(',', '').split(' ') 
                for i in xrange(len(words)):
                    if words[i] in vocab:
                        cluster_histogram[vocab.index(words[i])] += 1
                        total_occur += 1
            fread.close()

        if total_occur > 0:
            for m in xrange(vocab_size):
                cluster_histogram[m] /= float(total_occur)
        else:
            cluster_histogram.fill(1.0/vocab_size)

        line2 = str(cluster_histogram[0])
        for m in range(1, vocab_size):
            line2 += ';' + str(cluster_histogram[m])
        fw.write(line2 + '\n')
        fw.close()


    print "ASR features generated successfully!"
