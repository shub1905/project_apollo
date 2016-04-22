# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 00:02:44 2016

@author: patanjali
"""

import numpy

#DATA_DIR = '/home/patanjali/courses/4772/project/MillionSongSubset/data/'
DATA_DIR = '/home/patanjali/courses/4772/project/project_apollo/data/'

train_filename = 'mfcc_songs_10_train.npy'
valid_filename = 'mfcc_songs_10_valid.npy'
test_filename = 'mfcc_songs_10_test.npy'

def load_dataset():
    
    train = numpy.load(DATA_DIR + train_filename)
    validate = numpy.load(DATA_DIR + valid_filename)
    test = numpy.load(DATA_DIR + test_filename)
    
    return train, validate, test