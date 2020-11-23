# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:51:19 2020

@author: Yuchi
"""

import numpy as np
import warnings

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
        
def calc_Woodbury_R(r, last_R, n):
    '''
    Calculate the Correlation Matrix R use Woodbury’s identity
    
    param r: hyperspectral signal, type is  2d-array, shape is [band num, 1]
    param last_R: last R, R(n-1), type is 2d-array, shape is [band num, band num]
    param n: n-th point (now), type is int
    '''
    r = np.reshape(r, [-1, 1])
    v = 1/np.sqrt(n)*r
    vt = np.transpose(v)
    A = ((n-1)/n)*last_R
    try:
        Ainv = np.linalg.inv(A)
    except:
        Ainv = np.linalg.pinv(A)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in calc_Woodbury_R(), please check the input data')       
    
    new_Rinv = Ainv - (Ainv@v)@(vt@Ainv)/(1+vt@Ainv@v)
    return new_Rinv
    
    