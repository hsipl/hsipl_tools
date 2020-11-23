# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:44:42 2020

@author: Yuchi
"""

import numpy as np
import warnings
from . import calc_matrix as cm

def R_rxd(HIM, R = None, axis = ''):
    '''
    Reed–Xiaoli Detector for image to point use Correlation Matrix
    
    param HIM: hyperspectral imaging, type is 3d-array
    param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
    param axis: 'N' is normalized RXD, 'M' is modified RXD, other inputs represent the original RXD
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    rt = np.transpose(r)
    
    if R is None:
        R = cm.calc_R(HIM)
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in R_rxd(), please check the input data')
    
    if axis == 'N':
        n = np.sum((rt*rt), 1)
        result = np.sum(((np.dot(rt, Rinv))*rt), 1)
        result = result/n
    elif axis == 'M':
        n = np.power(np.sum((rt*rt), 1), 0.5)
        result = np.sum(((np.dot(rt, Rinv))*rt), 1)
        result = result/n
    else:
        result = np.sum(((np.dot(rt, Rinv))*rt), 1)
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def K_rxd(HIM, K = None, u = None, axis = ''):
    '''
    Reed–Xiaoli Detector for image to point use Covariance Matrix and mean value µ
    
    param HIM: hyperspectral imaging, type is 3d-array
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
    param axis: 'N' is normalized RXD, 'M' is modified RXD, other inputs represent the original RXD
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    if K is None or u is None:
        K, u = cm.calc_K_u(HIM) 
    ru = r-u
    rut = np.transpose(ru)
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in K_rxd(), please check the input data')
    
    if axis == 'N':
        n = np.sum((rut*rut), 1)
        result = np.sum(((np.dot(rut, Kinv))*rut), 1)
        result = result/n
    elif axis == 'M':
        n = np.power(np.sum((rut*rut), 1), 0.5)
        result = np.sum(((np.dot(rut, Kinv))*rut), 1)
        result = result/n
    else:
        result = np.sum((np.dot(rut, Kinv))*rut, 1)
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def CR_rxd(HIM, R = None):
    '''
    Reed–Xiaoli Detector for image to point use Causal Correlation Matrix
    用HIM算R，r是用最後一個點，return最後一個點的結果
    
    param HIM: hyperspectral imaging, type is 3d-array
    param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))[:, -1].reshape([HIM.shape[2], 1])
    rt = np.transpose(r)
    
    if R is None:
        R = cm.calc_R(HIM)
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in R_rxd(), please check the input data')
    result = rt@Rinv@r
    return result

def CK_rxd(HIM, K = None, u = None):
    '''
    Reed–Xiaoli Detector for image to point use Causal Covariance Matrix and mean value µ
    用HIM算K，r是用最後一個點，return最後一個點的結果
    
    param HIM: hyperspectral imaging, type is 3d-array
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))[:, -1].reshape([HIM.shape[2], 1])
    ru = r-u
    rut = np.transpose(ru)
    
    if K is None or u is None:
        K, u = cm.calc_K_u(HIM) 
    ru = r-u
    rut = np.transpose(ru)
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in K_rxd(), please check the input data')
    result = rut@Kinv@ru
    return result

def RT_CT_rxd(r, last_R, n):
    '''
    Reed–Xiaoli Detector for image to point use real-time Causal Correlation Matrix
    用Woodbuty算Rinv，r用最後一個點，return最後一個點的結果和R
    
    param r: hyperspectral signal, type is  2d-array, shape is [band num, 1]
    param last_R: last R, R(n-1), type is 2d-array, shape is [band num, band num]
    param n: n-th point (now), type is int
    '''
    r = np.reshape(r, [-1, 1])
    rt = np.transpose(r)
    Rinv = cm.calc_Woodbury_R(r, last_R, n)
    
    result = rt@Rinv@r
    R = np.linalg.inv(Rinv)
    return result, R
    
def utd(HIM, K = None, u = None):
    '''
    Uniform Target Detector for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    oneL = np.ones([1, HIM.shape[2]])
    if K is None or u is None:
        K, u = cm.calc_K_u(HIM)
    ru = r-u
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in utd(), please check the input data')
    
    result = (oneL-np.transpose(u))@Kinv@ru
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def rxd_utd(HIM, K = None, u = None):
    '''
    
    
    param HIM: hyperspectral imaging, type is 3d-array
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
    '''
    N = HIM.shape[0]*HIM.shape[1]
    r = np.transpose(HIM.reshape([N, HIM.shape[2]])) 
    if K is None or u is None:
        K, u = cm.calc_K_u(HIM)
    ru = r-u
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in rxd_utd(), please check the input data')

    result = np.sum((np.transpose(r-1)@Kinv)*np.transpose(ru), 1)
    result = result.reshape([HIM.shape[0], HIM.shape[1]])
    return result   

def lptd(HIM, R = None):
    '''
    Low Probability Target Detector
    
    param HIM: hyperspectral imaging, type is 3d-array
    param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    oneL = np.ones([1, HIM.shape[2]])
    if R is None:
        R = cm.calc_R(HIM) 
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in lptd(), please check the input data')
    
    result = np.dot(np.dot(oneL, Rinv), r)
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def sw_R_rxd(HIM, win):
    '''
    Sliding Window based Reed–Xiaoli Detector use Correlation Matrix
    Note: This function will cost a lot of CPU performance
    
    param HIM: hyperspectral imaging, type is 3d-array
    param win: window size used for sliding
    '''
    x, y, z = HIM.shape
    if win*2 > HIM.shape[0] or win*2 > HIM.shape[1]:
        raise ValueError('Wrong window size for sw_R_rxd()')
    half = np.fix(win / 2);
    result = np.zeros([x, y])
    
    for i in range(x):
        for j in range(y):
            x1 = i - half
            x2 = i + half
            y1 = j - half
            y2 = j + half
            
            if x1 <= 0:
                x1 = 0;
            elif x2 >= x:
                x2 = x
                
            if y1 <= 0:
                y1 = 0;
            elif y2 >= y:
                y2 = y
            
            x1 = np.int(x1)
            x2 = np.int(x2)
            y1 = np.int(y1)
            y2 = np.int(y2)
            
            Local_HIM = HIM[x1:x2, y1:y2, :]
            
            xx, yy, zz = Local_HIM.shape
            X = np.reshape(np.transpose(Local_HIM), (zz, xx*yy))
            S = np.dot(X, np.transpose(X))
            r = np.reshape(HIM[i, j, :], [z,1])
            
            IS = np.linalg.inv(S)
         
            result[i,j] = np.dot(np.dot(np.transpose(r), IS), r)
    
    return result