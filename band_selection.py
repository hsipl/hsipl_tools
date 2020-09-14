# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:43:27 2020

@author: Yuchi
"""
import numpy as np
from scipy.stats import entropy
import time
import pandas as pd

def CEM_BCC(imagecube, num):
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx, yy, band_num))
    test_image = np.mat(np.array(test_image))
    R = test_image*test_image.T/(xx*yy*1.0)
    
    tt = np.mat(R) ** -1
    
    score=np.zeros(( band_num,band_num))
    for i in range(0,band_num):
        endmember_matrix = test_image[:, i]
        W = tt* endmember_matrix * ((endmember_matrix.T * tt * endmember_matrix)**-1)
        for j in range(0,band_num):
            if i != j:
                test = test_image[:, j]
                score[i, j] = test.T * W
            else:
                score[i,j] = 1
    weight = np.zeros((band_num,1))
    for i in range(0,band_num):
        test = score[i,:]
        scalar = np.sum(test) - score[i,i]
        weight[i] = scalar
        
    weight = np.abs(weight)
    original = range(0,band_num)
    coefficient_integer = weight * 1
    band_select = np.zeros((num,1))
    
    sorted_indices = np.argsort(-coefficient_integer, axis=0)
    original=np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]

    band_sort = band_select.copy()
    band_sort = band_sort.sort()
    
    return band_select

def CEM_BCM(imagecube, num):
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num))
    test_image = np.mat(np.array(test_image))
    R = test_image*test_image.T/(xx*yy*1.0)
    tt = np.mat(R) ** -1
    score=np.zeros((band_num,1))
    for i in range(0,band_num):
        endmember_matrix = test_image[:, i]
        W = tt* endmember_matrix * ((endmember_matrix.T * tt * endmember_matrix)**-1)
        score[i] = W.T * R * W
    weight = np.abs(score)
    original = range(0,band_num)
    coefficient_integer = weight * 1
    band_select = np.zeros((num,1))
    
    sorted_indices = np.argsort(-coefficient_integer, axis=0)
    original=np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    band_sort = band_select.copy()
    band_sort = band_sort.sort()
    
    return band_select

def CEM_BDM(self, imagecube, num):
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num))
    test_image = np.mat(np.array(test_image))
    R = test_image*test_image.T/(xx*yy*1.0)
    
    score=np.zeros((band_num,1))
    for i in range(0,band_num):
        endmember_matrix = test_image[:, i]
        R_new = R - endmember_matrix * endmember_matrix.T
        R_new = R_new/(band_num -1)
        tt = np.mat(R_new) ** -1
        W = tt* endmember_matrix * ((endmember_matrix.T * tt * endmember_matrix)**-1)
        score[i] = W.T * R * W
    weight = np.abs(score)
    original = range(0,band_num)
    coefficient_integer = weight * 1
    band_select = np.zeros((num,1))
    
    sorted_indices = np.argsort(-coefficient_integer, axis=0)
    original=np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    band_sort = band_select.copy()
    band_sort = band_sort.sort()
    
    return band_select

def BS_STD(imagecube, num):
    hyperspectral = imagecube.reshape((imagecube.shape[0]*imagecube.shape[1],imagecube.shape[2] ))
    hyperspectral_std=np.zeros((2,imagecube.shape[2]))
    hyperspectral_std[0,:] =range(0,imagecube.shape[2])
    for i in range(0,imagecube.shape[2]):
        hyperspectral_std[1,i] =np.std(hyperspectral[:,i])
    
    for i in range(0,imagecube.shape[2]):
        for j in range(0,imagecube.shape[2]):
            if hyperspectral_std[1,i] > hyperspectral_std[1,j]:
                temp = hyperspectral_std[:,i].copy()
                hyperspectral_std[:,i]= hyperspectral_std[:,j].copy()
                hyperspectral_std[:,j]= temp.copy()
    newcube = np.zeros((imagecube.shape[0]*imagecube.shape[1],num))
    
    for i in range(0,imagecube.shape[0]*imagecube.shape[1]):
        for j in range(0,num):
            newcube[i,j]=hyperspectral[i,int(hyperspectral_std[0,j])]
    band_select = hyperspectral_std[0,:num]
    return newcube,band_select        

def BS_Corrcoef(imagecube, num):
    hyperspectral = imagecube.reshape((imagecube.shape[0]*imagecube.shape[1],imagecube.shape[2] ))
    scores = np.zeros((imagecube.shape[2],imagecube.shape[2]))
    for i in range(imagecube.shape[2]):
        for j in range(imagecube.shape[2]):
            scores[i,j] = np.min(np.min(np.corrcoef(hyperspectral[:,i],hyperspectral[:,j])))
    hyperspectral_corrcoef=np.zeros((2,imagecube.shape[2]))
    hyperspectral_corrcoef[0,:] = range(imagecube.shape[2])
    for i in range(imagecube.shape[2]):
        hyperspectral_corrcoef[1,i] = np.sum(scores[i,:])-scores[i,i]
    for i in range(0,imagecube.shape[2]):
        for j in range(0,imagecube.shape[2]):
            if hyperspectral_corrcoef[1,i] > hyperspectral_corrcoef[1,j]:
                temp = hyperspectral_corrcoef[:,i].copy()
                hyperspectral_corrcoef[:,i]= hyperspectral_corrcoef[:,j].copy()
                hyperspectral_corrcoef[:,j]= temp.copy()
    band_select = hyperspectral_corrcoef[0,:num]
                
    return band_select  
        
def BS_Entropy(imagecube, num):
    hyperspectral = imagecube.reshape((imagecube.shape[0]*imagecube.shape[1],imagecube.shape[2] ))
    hyperspectral_entropy = np.zeros((2,imagecube.shape[2]))
    hyperspectral_entropy[0,:]=range(imagecube.shape[2])
    
    for i in range(imagecube.shape[2]):
        hyperspectral_entropy[1,i] = entropy(hyperspectral[:,i])
    for i in range(0,imagecube.shape[2]):
        for j in range(0,imagecube.shape[2]):
            if hyperspectral_entropy[1,i] > hyperspectral_entropy[1,j]:
                temp = hyperspectral_entropy[:,i].copy()
                hyperspectral_entropy[:,i]= hyperspectral_entropy[:,j].copy()
                hyperspectral_entropy[:,j]= temp.copy()
    band_select = hyperspectral_entropy[0,:num]
                
    return band_select

def BS_minV_BP(imagecube, d, num):
    start = time.clock()
    X = []
    for i in range(d.shape[1]):
        dd = d[:, i].reshape((d.shape[0], 1), order='F')
        minV_BP_band_select, minV_BP_runtime = minV_BP(imagecube, dd, num)
        minV_BP_band_select = minV_BP_band_select.reshape((minV_BP_band_select.shape[0]), order='F')
        X.append(minV_BP_band_select)
      
    X = np.array(X)
    a = pd.Series(X.reshape((X.shape[0]*X.shape[1]), order='F'))
    b = a.value_counts()
    band_select = np.array(b[:num].index)
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime

def BS_maxV_BP(imagecube, d, num):
    start = time.clock()
    X = []
    for i in range(d.shape[1]):
        dd = d[:, i].reshape((d.shape[0], 1), order='F')
        maxV_BP_band_select, maxV_BP_runtime = maxV_BP(imagecube, dd, num)
        maxV_BP_band_select = maxV_BP_band_select.reshape((maxV_BP_band_select.shape[0]), order='F')
        X.append(maxV_BP_band_select)
      
    X = np.array(X)
    a = pd.Series(X.reshape((X.shape[0]*X.shape[1]), order='F'))
    b = a.value_counts()
    band_select = np.array(b[:num].index)
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime

def BS_SF_CTBS(imagecube, d, num):
    start = time.clock()
    X = []
    for i in range(d.shape[1]):
        dd = d[:, i].reshape((d.shape[0], 1), order='F')
        SF_CTBS_band_select, SF_CTBS_runtime = SF_CTBS(imagecube, dd, num)
        SF_CTBS_band_select = SF_CTBS_band_select.reshape((SF_CTBS_band_select.shape[0]), order='F')
        X.append(SF_CTBS_band_select)
      
    X = np.array(X)
    a = pd.Series(X.reshape((X.shape[0]*X.shape[1]), order='F'))
    b = a.value_counts()
    band_select = np.array(b[:num].index)
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime

def BS_SB_CTBS(imagecube, d, num):
    start = time.clock()
    X = []
    for i in range(d.shape[1]):
        dd = d[:, i].reshape((d.shape[0], 1), order='F')
        SB_CTBS_band_select, SB_CTBS_runtime = SB_CTBS(imagecube, dd, num)
        SB_CTBS_band_select = SB_CTBS_band_select.reshape((SB_CTBS_band_select.shape[0]), order='F')
        X.append(SB_CTBS_band_select)
      
    X = np.array(X)
    a = pd.Series(X.reshape((X.shape[0]*X.shape[1]), order='F'))
    b = a.value_counts()
    band_select = np.array(b[:num].index)
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime

def SF_CTBS(imagecube, d, num):
    start = time.clock()
    
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num), order='F')
    
    min_band_select, non = minV_BP(imagecube, d, num)
    
    omega = []
    omega.append(np.int(min_band_select[0]))
    
    for i in range(0, num-1):
        score = np.zeros((band_num,1))
        for j in range(0, band_num):
            new_d = []
            new_r = []
            bl = []
            bl.append(j)
            omega_bl = list(set(omega) | set(bl))
            
            for k in omega_bl:
                new_d.append(d[k])
                new_r.append(test_image[:, k])
                
            new_d = np.array(new_d)
            new_r = np.array(new_r)
            
            new_R = np.dot(new_r, np.transpose(new_r))/(xx*yy*1.0)
            new_tt = np.linalg.inv(new_R)
            new_W = 1 / (np.dot(np.dot(np.transpose(new_d), new_tt), new_d))
            score[j] = new_W
        weight = np.abs(score)
        coefficient_integer = weight * 1
        sorted_indices = np.argsort(coefficient_integer, axis=0)
        
        omega.append(np.int(sorted_indices[0]))
        
    band_select = np.array(omega)
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime

def SB_CTBS(imagecube, d, num):
    start = time.clock()
    
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num), order='F')
    
    min_band_select, non = maxV_BP(imagecube, d, num)
    
    omega = []
    omega.append(np.int(min_band_select[0]))
    
    for i in range(0, num-1):
        score = np.zeros((band_num,1))
        for j in range(0, band_num):
            bl = []
            bl.append(j)
            omega_bl = list(set(omega) | set(bl))
            
            new_d = np.delete(d, omega_bl, 0)
            new_r = np.delete(test_image, omega_bl, 1)
            
            new_R = np.dot(np.transpose(new_r), new_r)/(xx*yy*1.0)
            new_tt = np.linalg.inv(new_R)
            new_W = 1 / (np.dot(np.dot(np.transpose(new_d), new_tt), new_d))
            score[j] = new_W
        weight = np.abs(score)
        coefficient_integer = weight * -1
        sorted_indices = np.argsort(coefficient_integer, axis=0)
        
        omega.append(np.int(sorted_indices[0]))
        
    band_select = np.array(omega)
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime

def minV_BP(imagecube, d, num):
    start = time.clock()
    
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num), order='F')
    
    score=np.zeros((band_num,1))
    for i in range(0,band_num):
        r = test_image[:, i].reshape((test_image.shape[0], 1), order='F')
        R = np.dot(np.transpose(r), r)/(xx*yy*1.0)
        tt = np.linalg.inv(R)
        W =  1 / (np.dot(np.dot(np.transpose(d[i].reshape((1, 1), order='F')), tt), d[i].reshape((1, 1), order='F')))
        score[i] = W
    
    weight = np.abs(score)
    original = range(0,band_num)
    coefficient_integer = weight * 1
    band_select = np.zeros((num,1))
    
    sorted_indices = np.argsort(coefficient_integer, axis=0)
    original=np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime

def maxV_BP(imagecube, d, num):
    start = time.clock()
    
    xx,yy,band_num=imagecube.shape
    test_image = imagecube.reshape((xx*yy, band_num), order='F')
    
    score=np.zeros((band_num,1))
    for i in range(0,band_num):
        d_new = np.delete(d, i, 0)
        r = np.delete(test_image, i, 1)
        R = np.dot(np.transpose(r), r)/(xx*yy*1.0)
        tt = np.linalg.inv(R)
        W =  1 / ((np.dot(np.dot(np.transpose(d_new), tt), d_new)))
        score[i] = W
        
    weight = np.abs(score)
    original = range(0,band_num)
    coefficient_integer = weight * -1
    band_select = np.zeros((num,1))
    
    sorted_indices = np.argsort(coefficient_integer, axis=0)
    original=np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime

def uniform_BS(band_num, num):
    start = time.clock()
    
    score = np.random.uniform(0, 1, [band_num, 1])
    weight = np.abs(score)
    original = range(0,band_num)
    coefficient_integer = weight * -1
    band_select = np.zeros((num,1))
    
    sorted_indices = np.argsort(coefficient_integer, axis=0)
    original=np.array(original)
    sorted_y2 = original[sorted_indices]
    band_select = sorted_y2[:num]
    
    end = time.clock()
    runtime = end - start
    
    return band_select, runtime