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
    Reed–Xiaoli Detector for image to point use Correlation Matrix use HIM
    
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
    result = np.reshape(result, HIM.shape[:-1])
    return result

def K_rxd(HIM, K = None, u = None, axis = ''):
    '''
    Reed–Xiaoli Detector for image to point use Covariance Matrix and mean value µ use HIM
    
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
    result = np.reshape(result, HIM.shape[:-1])
    return result

def R_rxd_use_r(r, R = None):
    '''
    Reed–Xiaoli Detector for image to point use Correlation Matrix use r
    
    param r: hyperspectral signal, type is  2d-array, shape is [band num, point num]
    '''
    rt = np.transpose(r)
    if R is None:
        R = cm.calc_R_use_r(r)
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        
    result = np.sum(((np.dot(rt, Rinv))*rt), 1)
    return result

def K_rxd_use_r(r, K = None, u = None):
    '''
    Reed–Xiaoli Detector for image to point use Covariance Matrix and mean value µ use r
    
    param r: hyperspectral signal, type is  2d-array, shape is [band num, point num]
    '''
    if K is None or u is None:
        K, u = cm.calc_K_u_use_r(r) 
    ru = r-u
    rut = np.transpose(ru)
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
    result = np.sum((np.dot(rut, Kinv))*rut, 1)
    return result

def CR_rxd(HIM):
    '''
    Reed–Xiaoli Detector for image to point use Causal Correlation Matrix
    # R: 第1次是拿前169個點算，接著拿前170個點、前171個...
    # r: 第1次是拿前169個點，後面都是拿最新的1個點算
    
    param HIM: hyperspectral imaging, type is 3d-array
    '''
    rt = np.reshape(HIM, [-1, HIM.shape[2]])
    r = np.transpose(rt)
    
    L, N = r.shape
    R = cm.calc_R_use_r(r[:, :L])
    result = np.zeros(N)
    result[:L] = R_rxd_use_r(r[:, :L], R)
    
    for i in range(L, N):
        R = cm.calc_R_use_r(r[:, :i+1])
        result[i:i+1] = R_rxd_use_r(r[:, i:i+1], R)
    result = np.reshape(result, HIM.shape[:-1])
    return result

def CK_rxd(HIM):
    '''
    Reed–Xiaoli Detector for image to point use Causal Covariance Matrix and mean value µ
    # K: 第1次是拿前169個點算，接著拿前170個點、前171個...
    # r: 第1次是拿前169個點，後面都是拿最新的1個點算
    
    param HIM: hyperspectral imaging, type is 3d-array
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    
    L, N = r.shape
    K, u = cm.calc_K_u_use_r(r[:, :L])
    result = np.zeros(N)
    result[:L] = K_rxd_use_r(r[:, :L], K, u)
    
    for i in range(L, N):
        K, u = cm.calc_K_u_use_r(r[:, :i+1])
        result[i:i+1] = K_rxd_use_r(r[:, i:i+1], K, u)
    result = np.reshape(result, HIM.shape[:-1])
    return result

def RT_CR_rxd(HIM):
    '''
    Reed–Xiaoli Detector for image to point use real-time Causal Correlation Matrix
    # R: 第1個R拿前169個點算，第n個R拿第n-1個R用Woodbury推算
    # r: 第1次是拿前169個點，後面都是拿最新的1個點算
    
    param HIM: hyperspectral imaging, type is 3d-array
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    
    L, N = r.shape
    result = np.zeros(N)
    R = cm.calc_R_use_r(r[:, :L])
    result[:L] = R_rxd_use_r(r[:, :L], R)
    
    for i in range(L, N):
        Rinv = cm.calc_Woodbury_R(r[:, i:i+1], R, i+1)
            
        rrt = r[:, i:i+1].T
        result[i:i+1] = np.sum(((np.dot(rrt, Rinv))*rrt), 1)  
        R = np.linalg.inv(Rinv)
        
    result = np.reshape(result, HIM.shape[:-1])
    return result

def RT_CK_rxd(HIM):
    '''
    Reed–Xiaoli Detector for image to point use real-time Causal Covariance Matrix and mean value µ
    # K: 第1個R拿前169個點算，第n個R拿第n-1個R用Woodbury推算
    # r: 第1次是拿前169個點，後面都是拿最新的1個點算
    
    param HIM: hyperspectral imaging, type is 3d-array
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    
    L, N = r.shape
    result = np.zeros(N)
    K, u = cm.calc_K_u_use_r(r[:, :L])
    result[:L] = K_rxd_use_r(r[:, :L], K, u)
    
    for i in range(L, N):
        Kinv, u = cm.calc_Woodbury_K_u(r[:, i:i+1], K, u, i+1)
        
        rut = np.transpose(r[:, i:i+1]-u)
        result[i:i+1] = np.sum((np.dot(rut, Kinv))*rut, 1)
        K = np.linalg.inv(Kinv)        
        
    result = np.reshape(result, HIM.shape[:-1])
    return result
    
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
    result = np.reshape(result, HIM.shape[:-1])
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
    result = np.reshape(result, HIM.shape[:-1])
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
    result = np.reshape(result, HIM.shape[:-1])
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