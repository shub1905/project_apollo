# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 23:51:12 2016

@author: patanjali
"""

from sklearn.linear_model import MultiTaskElasticNetCV
from utils2 import load_dataset
import pandas

train, validate, test = load_dataset()

no_classes = train[:,0].max()+1
train_y = pandas.get_dummies(train[:,0])

print no_classes, train.shape

train = train[:201]
validate = validate[:201]
test = test[:201]

for l1_ratio in [.1, .5, .7, .9, .95, .99, 1]:
    
    model = MultiTaskElasticNetCV(l1_ratio=l1_ratio, normalize=True, verbose=True, n_jobs=3)
    model.fit(train[:,1:], train_y)
    predicted_classes = (model.predict(validate[:,1:])).argmax(1)
    
    correct = sum(predicted_classes==validate[:,0])
    print l1_ratio, correct, correct*1.0/validate.shape[0]
    