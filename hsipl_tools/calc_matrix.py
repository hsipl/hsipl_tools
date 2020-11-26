# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:51:19 2020

@author: Yuchi
"""

import numpy as np
import warnings

def calc_R(HIM):
    '''
    Calculate the Correlation Matrix R use HIM
    
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
    Calculate the Covariance Matrix K and mean value µ use HIM
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
        
def calc_R_use_r(r):
    '''
    Calculate the Correlation Matrix R use r
    
    param r: hyperspectral signal, type is  2d-array, shape is [band num, point num]
    '''
    try:
        if np.ndim(r) == 1:  # 如果r的形狀是[n, ]，就幫他變成[n, 1]
            N = 1
            r = np.reshape(r, [-1, 1])
        else:
            N = r.shape[1]
        R = 1/N*(r@r.T)
        return R
    except:
        print('An error occurred in calc_R_use_r()')

def calc_K_u_use_r(r):
    '''
    Calculate the Covariance Matrix K and mean value µ use r
    
    param r: hyperspectral signal, type is  2d-array, shape is [band num, point num]
    '''
    try:
        if len(r.shape) == 1:  # 如果r的形狀是[n, ]，就幫他變成[n, 1]
            N = 1
            r = np.expand_dims(r, 1)
        else:
            N = r.shape[1]
        u = (np.mean(r, 1)).reshape(r.shape[0], 1)        
        K = 1/N*np.dot(r-u, np.transpose(r-u))
        return K, u
    except:
        print('An error occurred in calc_K_u_use_r()')
         
def calc_Woodbury_R(r, last_R, n):
    '''
    Calculate the Correlation Matrix R use Woodbury’s identity
    
    param r: hyperspectral signal, type is  2d-array, shape is [band num, 1]
    param last_R: last Correlation Matrix, R(n-1), type is 2d-array, shape is [band num, band num]
    param n: n-th point (now), type is int
    '''
    A = ((n-1)/n)*last_R
    u = 1/np.sqrt(n)*r
    v = u
    vt = np.transpose(v)
    try:
        Ainv = np.linalg.inv(A)
    except:
        Ainv = np.linalg.pinv(A)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in calc_Woodbury_R(), please check the input data')
    
    new_Rinv = Ainv - (Ainv@u)@(vt@Ainv)/(1+vt@Ainv@u)
    return new_Rinv
    
def calc_Woodbury_K_ru(r, last_K, last_u, n):
    '''
    Calculate the Covariance Matrix K and r-µ use Woodbury’s identity
    
    param r: hyperspectral signal, type is  2d-array, shape is [band num, 1]
    param last_K: last Covariance Matrix, K(n-1), type is 2d-array, shape is [band num, band num]
    param last_u: last mean value µ, u(n-1), shape is [band num, 1]
    param n: n-th point (now), type is int
    '''
    ru = r-(1-1/n)*last_u+(1/n)*r
    A = ((n-1)/n)*last_K
    u = np.sqrt(n-1)/n*(last_u-r)-r
    vt = np.transpose(u)
    try:
        Ainv = np.linalg.inv(A)
    except:
        Ainv = np.linalg.pinv(A)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in calc_Woodbury_K_ru(), please check the input data')    
    
    new_Kinv = Ainv - (Ainv@u)@(vt@Ainv)/(1+vt@Ainv@u)
    return new_Kinv, ru
    