# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 18:55:27 2020

@author: Yuchi
"""

import numpy as np
from skimage import segmentation, measure, filters
def data_normalize(input_data):
    input_data = np.array(input_data)*1.0
    maximum = np.max(np.max(input_data))
    minimum = np.min(np.min(input_data))
    
    normal_data = (input_data-minimum)/(maximum-minimum)*1.0
    return normal_data

def msc(input_data, reference = None):
    ''' 
    Multiplicative Scatter Correction
    
    param input_data: signal, type is 2d-array, size is [signal num, band num]
    param reference: reference of msc, type is 2d-array, size is [1, band num], if reference is None, it will be calculated in the function
    '''
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i,:].mean()
    if reference is None:
        ref = np.mean(input_data, axis = 0)
    else:
        ref = reference
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        fit = np.polyfit(ref, input_data[i,:], 1, full=True)
        data_msc[i,:] = (input_data[i,:] - fit[0][1]) / fit[0][0] 
    return (data_msc, ref)
   
def snv(input_data):
    '''
    Standard Normal Variate
    
    param input_data: signal, type is 2d-array, size is [signal num, band num]
    '''
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        data_snv[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return data_snv

def awgn(x, snr):
    '''
    Additive white Gaussian noise 高斯白雜訊
    
    param x: signal, type is 2d-array, size is [band num, 1]
    '''
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise
def dust_roi(img, mask,ignoreArea=10):
    try:
        label_roi = []
        cleared = mask.copy()
        segmentation.clear_border(cleared)
        label_image = measure.label(cleared)
        borders = np.logical_xor(mask, cleared)
        label_image[borders] = -1
        x = 5
        for region in measure.regionprops(label_image):
            # 忽略小區域
            if region.area < ignoreArea:
                continue
            # ROI
            minr, minc, maxr, maxc = region.bbox
            label_roi.append(img[minr-x:maxr+x, minc-x:maxc+x, :])
    except Exception as e:
        print(e)
    else:
        return label_roi

