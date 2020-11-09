# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:51:19 2020

@author: Yuchi
"""

import numpy as np

def calc_R(HIM):
    '''
    Calculate the Correlation Matrix R
    
    param HIM: hyperspectral imaging, type is 3d-array
    '''
    try:
        N = HIM.shape[0]*HIM.shape[1]
        r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
        R = 1/N*(r@r.T)
        return R
    except:
        print('An error occurred in calc_R()')

def calc_K_u(HIM):
    '''
    Calculate the Covariance Matrix K and mean value µ
    mean value µ was named u just because it looks like u :P
    
    param HIM: hyperspectral imaging, type is 3d-array
    '''
    try:
        N = HIM.shape[0]*HIM.shape[1]
        r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
        u = (np.mean(r, 1)).reshape(HIM.shape[2], 1)        
        K = 1/N*np.dot(r-u, np.transpose(r-u))
        return K, u
    except:
        print('An error occurred in calc_K_u()')