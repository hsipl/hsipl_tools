# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 19:59:38 2020

@author: Yuchi
"""

import numpy as np
import warnings
import calc_matrix as cm

def cem(HIM, d, R = None):
    '''
    Constrained Energy Minimization for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
    '''  
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    if R is None:
        R = cm.calc_R(HIM)
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in cem_img(), please check the input data')
    result = np.dot(np.transpose(r), np.dot(Rinv, d))/np.dot(np.transpose(d), np.dot(Rinv, d))
    result = np.reshape(result, [HIM.shape[0], HIM.shape[1]])
    return result

def subset_cem(HIM, d, win_height = None, win_width = None):
    '''
    Subset Constrained Energy Minimization
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param win_height: window height for subset cem, type is int
    param win_width: window width for subset cem, type is int
    '''
    if win_height is None:
        win_height = np.ceil(HIM.shape[0] / 3)
    
    if win_width is None:
        win_width = np.ceil(HIM.shape[1] / 3)
    
    if win_height > HIM.shape[0] or win_width > HIM.shape[1] or win_height < 2 or win_width < 2:
        raise ValueError('Wrong window size for subset_cem()')
    
    d = np.reshape(d, [HIM.shape[2], 1])
    result = np.zeros([HIM.shape[0], HIM.shape[1]])

    for i in range(0, HIM.shape[0], win_height):
        for j in range(0, HIM.shape[1], win_width):
            result[i: i + win_height, j: j + win_width] = cem(HIM[i: i + win_height, j: j + win_width, :], d)
    return result

def sw_cem(HIM, d, win):
    '''
    Sliding Window based Constrained Energy Minimization
    Note: This function will cost a lot of CPU performance
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param win: window size used for sliding
    '''
    if win*2 > HIM.shape[0] or win*2 > HIM.shape[1]:
        raise ValueError('Wrong window size for sw_cem()')
    half = np.fix(win/2);
    result = np.zeros([HIM.shape[0], HIM.shape[1]])

    for i in range(HIM.shape[0]):
        for j in range(HIM.shape[1]):
            x1 = i - half
            x2 = i + half
            y1 = j - half
            y2 = j + half 
        
            if x1 <= 0:
                x1 = 0;
            elif x2 >= HIM.shape[0]:
                x2 = HIM.shape[0]
            
            if y1 <= 0:
                y1 = 0;
            elif y2 >= HIM.shape[1]:
                y2 = HIM.shape[1]
        
            x1 = np.int(x1)
            x2 = np.int(x2)
            y1 = np.int(y1)
            y2 = np.int(y2)
        
            Local_HIM = HIM[x1:x2, y1:y2, :]
            X = np.reshape(np.transpose(Local_HIM), (Local_HIM.shape[2], Local_HIM.shape[0]*Local_HIM.shape[1]))
            S = np.dot(X,np.transpose(X))
            r = np.reshape(HIM[i,j,:], [HIM.shape[2], 1])
        
            S = np.linalg.inv(S)
     
            result[i,j] = np.dot(np.dot(np.transpose(r), S), d) / np.dot(np.dot(np.transpose(d), S), d)

    return result

def hcem(HIM, d, max_it = 100, λ = 200, e = 1e-6):
    '''
    Hierarchical Constrained Energy Minimization
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param max_it: maximum number of iterations, type is int
    param λ: coefficients for constructing a new CEM detector, type is int
    param e: stop iterating until the error is less than e, type is int
    '''
    imgH = HIM.shape[0]  #image height
    imgW = HIM.shape[1]  #image width
    N = imgH*imgW  #pixel number
    D = HIM.shape[2]  #band number
    Weight = np.ones([1, N])
    y_old = np.ones([1, N])
    
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    
    for T in range(max_it):
        for pxlID in range(N):
            r[:, pxlID] = r[:, pxlID]*Weight[:, pxlID]
        R = r@r.T / N
        
        w = np.linalg.inv(R + 0.0001*np.eye(D))@d / (d.T@np.linalg.inv(R + 0.0001*np.eye(D))@d)
        
        y = w.T@r
        Weight = 1 - np.exp(-λ*y)
        Weight[Weight < 0] = 0
        
        res = np.linalg.norm(y_old)**2/N - np.linalg.norm(y)**2/N
        print(f'iteration {T + 1}: ε = {res}')
        y_old = y.copy()
        
        #stop criterion:
        if np.abs(res) < e:
             break;
             
        #display the detection results of each layer 
        hCEMMap = np.reshape(y, [imgH, imgW])
    return hCEMMap

def sam_img(HIM, d):
    '''
    Spectral Angle Match for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    
    rr = np.sum(r**2, 0)**0.5
    dd = np.sum(d**2, 0)**0.5
    rd = np.sum(r*d, 0)
    result = np.arccos(rd/(rr*dd))
    
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result
    
def sam_point(p1, p2):
    '''
    Spectral Angle Match for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point same as d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    rd = np.sum((p1*p2), 0)
    rr = np.power(np.sum(np.power(p1, 2), 0), 0.5)
    dd = np.power(np.sum(np.power(p2, 2), 0), 0.5)
    x = rd/(rr*dd)
    result = np.arccos(abs(x))
    return result

def ed_img(HIM, d):
    '''
    Euclidean Distance for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])      
    
    result = np.sum((r-d)**2, 0)**0.5
    result = result.reshape(HIM.shape[0], HIM.shape[1])                    
    return result

def ed_point(p1, p2):
    '''
    Euclidean Distance for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point same as d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    x = p1-p2
    result = np.dot(np.transpose(x), x)
    return result

def sid_img(HIM, d):
    '''
    Spectral Information Divergence for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    
    m = r/np.sum(r, 0)
    n = d/np.sum(d, 0)        
    drd = np.sum(m*np.log(m/n), 0)
    ddr = np.sum(n*np.log(n/m), 0)
    result = drd+ddr
    
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def sid_point(p1, p2):
    '''
    Spectral Information Divergence for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point same as d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    m = p1/np.sum(p1, 0)
    n = p2/np.sum(p2, 0)
    x = np.sum(m*np.log(m/n), 0)
    y = np.sum(n*np.log(n/m), 0)
    result = x+y
    return result

def sid_tan(p1, p2):
    '''
    SID-SAM Mixed Measure for point to point
    SID(TAN) = SID(s, s') x tan(SAM(s, s'))
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point same as d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    sid_res = sid_point(p1, p2)
    sam_res = sam_point(p1, p2)
    result = sid_res * np.tan(sam_res)
    return result

def sid_sin(p1, p2):
    '''
    SID-SAM Mixed Measure for point to point
    SID(SIN) = SID(s, s') x sin(SAM(s, s'))
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point same as d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    sid_res = sid_point(p1, p2)
    sam_res = sam_point(p1, p2)
    result = sid_res * np.sin(sam_res)
    return result

def rsdpw_sam(p1, p2, d):
    '''
    Relative Spectral Discriminatory Power (RSDPW) with SAM for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    sam_p1d_res = sam_point(p1, d)
    sam_p2d_res = sam_point(p2, d)
    result = max(sam_p1d_res/sam_p2d_res, sam_p2d_res/sam_p1d_res)
    return result

def rsdpw_sid(p1, p2, d):
    '''
    Relative Spectral Discriminatory Power (RSDPW) with SID for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    sid_p1d_res = sid_point(p1, d)
    sid_p2d_res = sid_point(p2, d)
    result = max(sid_p1d_res/sid_p2d_res, sid_p2d_res/sid_p1d_res)
    return result

def rsdpw_sid_tan(p1, p2, d):
    '''
    Relative Spectral Discriminatory Power (RSDPW) with SID(TAN) for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    sid_tan_p1d_res = sid_tan(p1, d)
    sid_tan_p2d_res = sid_tan(p2, d)
    result = max(sid_tan_p1d_res/sid_tan_p2d_res, sid_tan_p2d_res/sid_tan_p1d_res)
    return result

def rsdpw_sid_sin(p1, p2, d):
    '''
    Relative Spectral Discriminatory Power (RSDPW) with SID(SIN) for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    sid_sin_p1d_res = sid_sin(p1, d)
    sid_sin_p2d_res = sid_sin(p2, d)
    result = max(sid_sin_p1d_res/sid_sin_p2d_res, sid_sin_p2d_res/sid_sin_p1d_res)
    return result

def ace(HIM, d, K = None, u = None):
    '''
    Adaptive Cosin/Coherent Estimator for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]    
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])    
    if K is None or u is None:
        K, u = cm.calc_K_u(HIM)
    rt = np.transpose(r)
    dt = np.transpose(d)
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
    
    result = (dt@Kinv@r)**2 / ((dt@Kinv@d) * np.sum((rt@Kinv)*rt, 1)).reshape(1, HIM.shape[0]*HIM.shape[1])
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def mf(HIM, d, K = None, u = None):
    '''
    Matched Filter for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    if K is None or u is None:
        K, u = cm.calc_K_u(HIM)
    du = d-u
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
        
    k = 1 / (du.T@Kinv@du)    
    w = k*(Kinv@du)
    result = w.T@r
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

#def glrt(HIM, d, non_d):
#    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
#    rt = np.transpose(r)
#    pr = np.eye(non_d.shape[0]) - non_d@(non_d.T@non_d)@non_d.T
#    dU = np.hstack([d, non_d])
#    psr = np.eye(non_d.shape[0]) - dU@(dU.T@dU)@dU.T
#    gl = np.sum(rt@(pr-psr)*rt, 1) / np.sum((rt@psr)*rt, 1)
#    result = np.reshape(gl, [HIM.shape[0], HIM.shape[1]])
#    return result

def kmd_img(HIM, d, K = None, u = None):
    '''
    Covariance Mahalanobis Distance for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    if K is None or u is None:
        K, u = cm.calc_K_u(HIM)
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in kmd_img(), please check the input data')
        
    result = np.sum(Kinv@(r-d)*(r-d), 0)**0.5
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def kmd_point(p1, p2, K = None):
    '''
    Covariance Mahalanobis Distance for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point same as d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    '''
    if K is None:
        K, _ = cm.calc_K_u(np.expand_dims(p1, 0))
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in kmd_point(), please check the input data')
    x = p1-p2
    result = np.dot(np.dot(np.transpose(x), Kinv), x)
    return result

def rmd_img(HIM, d, R = None):
    '''
    Correlation Mahalanobis Distance for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    if R is None:
        R = cm.calc_R(HIM)    
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in rmd_img(), please check the input data')
        
    result = np.sum(Rinv@(r-d)*(r-d), 0)**0.5
    result = 1-(result/np.max(result))
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def rmd_point(p1, p2, R = None):
    '''
    Correlation Mahalanobis Distance for point to point
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point same as d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
    '''
    if R is None:
        R = cm.calc_R(np.expand_dims(p1, 0))
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        warnings.warn('The pseudo-inverse matrix is used instead of the inverse matrix in rmd_point(), please check the input data')
    x = p1-p2
    result = np.dot(np.dot(np.transpose(x), Rinv), x)
    return result

def kmfd(HIM, d, K = None, u = None):
    '''
    Covariance Matched Filter based Distance for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param K: Covariance Matrix, type is 2d-array, if K is None, it will be calculated in the function
    param u: mean value µ, type is 2d-array, size is same as param d, if u is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    if K is None or u is None:
        K, u = cm.calc_K_u(HIM)
    try:
        Kinv = np.linalg.inv(K)
    except:
        Kinv = np.linalg.pinv(K)
        
    result = (r-u).T@Kinv@(d-u)  
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def rmfd(HIM, d, R = None):
    '''
    Correlation Matched Filter based Distance for image to point
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    param R: Correlation Matrix, type is 2d-array, if R is None, it will be calculated in the function
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    if R is None:
        R = cm.calc_R(HIM)    
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
        
    result = r.T@Rinv@d
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def tcimf(HIM, d, no_d):
    '''
    Target-Constrained Interference-Minimized Filter
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, point num], for example: [224, 1], [224, 3]
    param no_d: undesired target, type is 2d-array, size is [band num, point num], for example: [224, 1], [224, 3]
    '''
    r = np.transpose(np.reshape(HIM, [-1, HIM.shape[2]]))
    result = np.zeros([HIM.shape[0]*HIM.shape[1], 1])
    DU = np.hstack(([d, no_d]))
    d_count = d.shape[1]
    no_d_count = no_d.shape[1]
    DUtw = np.zeros([d_count + no_d_count, 1])
    DUtw[0: d_count] = 1
    R = (1/HIM.shape[0]*HIM.shape[1])*np.dot(r, np.transpose(r))
    try:
        Rinv = np.linalg.inv(R)
    except:
        Rinv = np.linalg.pinv(R)
    x = np.dot(np.dot(np.transpose(r), Rinv), DU)
    y = np.dot(np.dot(np.transpose(DU), Rinv), DU)
    y = np.linalg.inv(y)
    result = np.dot(np.dot(x, y), DUtw)  
    result = result.reshape(HIM.shape[0], HIM.shape[1])
    return result

def cbd_img(HIM, d):
    '''
    
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    r = np.transpose(np.reshape(HIM, [HIM.shape[0]*HIM.shape[1], HIM.shape[2]]))
    d = np.reshape(d, [HIM.shape[2], 1])
    
    result = np.sum(abs(r-d))
    result = result.reshape(HIM.shape[0], HIM.shape[1])   
    return result

def cbd_point(p1, p2):
    '''
    
    
    param p1: a point, type is 2d-array, size is [band num, 1], for example: [224, 1]
    param p2: a point same as d (Desired Signature), type is 2d-array, size is [band num, 1], for example: [224, 1]
    '''
    result = np.sum(abs(p1-p2), 0)
    return result
        
def LSOSP(HIM, d, no_d):
    '''
    Least Squares Orthogonal Subspace Projection
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, point num], for example: [224, 1], [224, 3]
    param no_d: undesired target, type is 2d-array, size is [band num, point num], for example: [224, 1], [224, 3]
    '''
    x, y, z = HIM.shape
    
    B = np.reshape(np.transpose(HIM), (z, x*y))
    I = np.eye(z)
    
    P = I - np.dot(np.dot(no_d, (np.linalg.inv(np.dot(np.transpose(no_d), no_d)))), np.transpose(no_d))
    
    lsosp = (np.dot(d.transpose(), P)) / (np.dot(np.dot(d.transpose(), P), d))
    
    dr = np.dot(lsosp, B)
    
    result = np.transpose(np.reshape(dr, [y, x]))
    
    return result

def KLSOSP(HIM, d, no_d, sig):
    '''
    Kernel-based Least Squares Orthogonal Subspace Projection
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, point num], for example: [224, 1]
    param no_d: undesired target, type is 2d-array, size is [band num, point num], for example: [224, 1]
    param sig: int, you can try 1, 10, 100, the results may all be different.
    
    KLSOSP是屬於kernael型的OSP 採用的kernael是RBF
    '''
    x, y, z = HIM.shape
    
    result = np.zeros([x, y])

    KdU = kernelized(d, no_d, sig)
    KUU = kernelized(no_d, no_d, sig)
    IKUU = np.linalg.inv(KUU)
    
    for i in range(x):
        for j in range(y):
            r = HIM[i, j, :].reshape(z, 1)
            
            Kdr = kernelized(d, r, sig)
            KUr = kernelized(no_d, r, sig)
            
            result[i, j] = Kdr - np.dot(np.dot(KdU, IKUU), KUr)          
    return result

def kernelized(x, y, sig):
    x_1, y_1 = x.shape
    x_2, y_2 = y.shape
    
    result = np.zeros([y_1, y_2])
    
    for i in range(y_1):
        for j in range(y_2):
            result[i, j] = np.exp((-1/2) * np.power(np.linalg.norm(x[:, i] - y[:, j]), 2) / (np.power(sig, 2)))
    
    return result

def AMSD(HIM, d, no_d):
    '''
    Adaptive Matched Subspace Detector
    
    param HIM: hyperspectral imaging, type is 3d-array
    param d: desired target d (Desired Signature), type is 2d-array, size is [band num, point num], for example: [224, 1]
    param no_d: undesired target, type is 2d-array, size is [band num, point num], for example: [224, 1]
    '''
    x, y, z = HIM.shape
    
    B = np.reshape(np.transpose(HIM), (z, x*y))

    I = np.eye(z)
    
    E = np.hstack([d, no_d])
    
    P_B = I - (np.dot(no_d, np.linalg.pinv(no_d)))
    
    P_Z = I - (np.dot(E, np.linalg.pinv(E)))
    
    tmp = P_B - P_Z
    
    dr = (np.sum(np.dot(B.transpose(), tmp) * B.transpose(), 1)) / (np.sum(np.dot(B.transpose(), P_Z) * B.transpose(), 1))
    
    result = np.transpose(np.reshape(dr, [y, x]))
    
    return result

def AMF(original, target):
    x, y, z = original.shape

    B = original.reshape(x * y, z)
    u = np.mean(B, 0)
    rep_u = u.reshape(1, z)
    
    Bu = B - rep_u
    
    K = np.dot(np.transpose(Bu), Bu) / (x * y)
    
    iK = np.linalg.inv(K)
    
    dr = np.dot(np.dot(target.transpose(), iK), B.transpose()) / np.dot(np.dot(target.transpose(), iK), target)
    
    AMF_result = dr.reshape(x, y)
    
    return AMF_result

def ASW_CEM(HIM, d, Sprout_HIM, minwd, midwd, maxwd, wd_range, sprout_rate):
    x, y, z = HIM.shape
    
    wd_matrix = np.zeros([x, y])
    mid_ASW_CEM_result = np.zeros([x, y])
    K = midwd
    
    for i in range(x):
        j = 0
        countnum = 0
        while j < y:
            half = np.fix(K / 2)
            
            x1 = np.int(i - half)
            x2 = np.int(i + half)
            y1 = np.int(j - half)
            y2 = np.int(j + half)
            
            if x1 <= 0:
                x1 = 0
            elif x2 >= x:
                x2 = x
                
            if y1 <= 0:
                y1 = 0
            elif y2 >= y:
                y2 = y
            
            xx, yy, zz = Sprout_HIM.shape
            Sprout = Sprout_HIM.reshape(xx * yy, zz)
            temp = np.sum(np.sum(Sprout[x1:x2, y1:y2], 0), 0)
            sumsprout = temp
            num = Sprout[x1:x2, y1:y2].shape[0] * Sprout[x1:x2, y1:y2].shape[1]
            
            if (sumsprout / num) < (sprout_rate - 0.001) and countnum == 0:
                K = K - wd_range
                j = j - 1
                countnum = 1
            elif (sumsprout / num) > (sprout_rate + 0.001) and countnum == 0:
                K = K + wd_range
                j = j - 1
                countnum = 2;
            elif (sumsprout / num) < (0.01) and countnum == 1 and K > minwd:
                K = K - wd_range
                j = j - 1
            elif (sumsprout / num) > (0.01) and countnum == 2 and K < maxwd:
                K = K + wd_range
                j = j - 1
            else:
                Local_HIM = HIM[x1:x2, y1:y2,:]
                
                xxx, yyy, zzz = Local_HIM.shape
                X = Local_HIM.reshape(xxx * yyy, zzz)
                S = np.dot(np.transpose(X), X)
                r = np.reshape(HIM[i, j, :], [z, 1])
				
                IS = np.linalg.inv(S)
                
                mid_ASW_CEM_result[i, j] = np.dot(np.dot(np.transpose(r), IS), d) / np.dot(np.dot(np.transpose(d), IS), d)
                wd_matrix[i, j] = K
                K = midwd
                countnum = 0
            j = j + 1
    
    return mid_ASW_CEM_result