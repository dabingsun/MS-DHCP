# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 09:19:08 2019

@author: dabing
"""

import numpy as np

def extractminibatch(kk,minibatch_num,batchsize,data):
    batch_start = minibatch_num *  batchsize
    batch_end = (minibatch_num+1) * batchsize
    n_samples = data.shape[0]
    if (batch_end + batchsize) <= n_samples:
        idx = kk[batch_start:batch_end]
        batch = data[idx]
    else:
        batch = data[kk[batch_start:]]
        
    return batch