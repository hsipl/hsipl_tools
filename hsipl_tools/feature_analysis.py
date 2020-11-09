# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:58:15 2020

@author: Yuchi
"""

import numpy as np
import scipy

def HFC(HIM, t):
    '''
    Harsanyi, Farrand, and Chang developed a NeymanPearson detection theory-based thresholding method (HFC)
    
    param HIM: hyperspectral imaging, type is 3d-array
    param t: a small number, you can try 1e-4, 1e-6 
    
    return band number
    '''
    x, y, z = HIM.shape
    pxl_no = x*y
    r = np.reshape(np.transpose(HIM), (z, x*y))
    
    R = np.dot(r, np.transpose(r)) / pxl_no
    u = (np.mean(r, 1)).reshape(z, 1)
    K = R - np.dot(u, np.transpose(u))
    
    D1 = np.linalg.eig(R)
    D1 = np.sort(D1[0], 0)
    
    D2 = np.linalg.eig(K)
    D2 = np.sort(D2[0], 0)
    
    sita = np.sqrt(((D1 ** 2 + D2 ** 2) * 2) / pxl_no)
    
    P_fa = t
    
    Threshold = (np.sqrt(2)) * sita * scipy.special.erfinv(1 - 2 * P_fa)
    
    Result = np.zeros([z, 1])
    
    for i in range(z):
        if (D1[i] - D2[i]) > Threshold[i]:
            Result[i] = 1
            
    number = int(np.sum(Result, 0))
    
    print('The VD number estimated is ' + str(number))
    
    return number

def NWHFC(HIM, t):
    '''
    Harsanyi, Farrand, and Chang developed a NeymanPearson detection theory-based thresholding method (HFC)
    Noise-Whitened HFC
    
    param HIM: hyperspectral imaging, type is 3d-array
    param t: a small number, you can try 1e-4, 1e-6 
    
    VD: Virtual Dimensionality
    
    return band number
    '''
    x, y, z = HIM.shape
    
    pxl_no = x*y
    r = np.reshape(np.transpose(HIM), (z, x*y))
    
    R = np.dot(r, np.transpose(r)) / pxl_no
    u = (np.mean(r, 1)).reshape(z, 1)
    K = R - np.dot(u, np.transpose(u))
    
    K_Inverse = np.linalg.inv(K)
    
    tuta = np.diag(K_Inverse)
    
    K_noise = 1 / tuta
    
    K_noise = np.diag(K_noise)
    
    image = np.dot(np.linalg.inv(scipy.linalg.sqrtm(K_noise)), r)
    
    image = np.transpose(np.reshape(image, (z, y, x)))
    
    number = HFC(image, t)
    
    print('The VD number estimated is ' + str(number))
    
    return number