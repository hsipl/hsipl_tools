# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:49:00 2020

@author: Yuchi
"""

import numpy as np
import random
import math

def ATGP(HIM, num_targets):
    xx = HIM.shape[0]
    yy = HIM.shape[1]
    bnd = HIM.shape[2]
    
    Loc = np.zeros([num_targets, 2])
    Sig = np.zeros([bnd, num_targets])
    
    r = np.reshape(np.transpose(HIM), [bnd, xx * yy])
    
    temp = np.sum((r * r), 0).reshape(1, xx * yy)
    
    a = np.max(temp)
    
    b = np.argwhere(temp == a)
    
    b = b[:, 1]
    
    if np.remainder(b, xx) == 0:
        Loc[0, 0] = b / xx
        Loc[0, 1] = xx
    elif np.floor(b / xx) == 0:
        Loc[0, 0] = 0
        Loc[0, 1] = b
    else:
        Loc[0, 0] = np.floor(b / xx)
        Loc[0, 1] = b - xx * np.floor(b / xx)
    
    Sig[:, 0] = r[:, b].reshape(bnd)
    
    for m in range(num_targets - 1):
        U = Sig[:, 0:m + 1]
        
        c = np.dot(np.transpose(U), U)
        
        if c.shape[0] == 1:
            c = 1 / c
        else:
            c = np.linalg.inv(c)
        
        P_U_perl = np.eye(bnd) - np.dot(np.dot(U, c), np.transpose(U))
        y = np.dot(P_U_perl,r)
        temp = np.sum((y * y), 0).reshape(1, xx * yy)
        a = np.max(temp)
        
        b = np.argwhere(temp == a)
        print(b)
        print('\n')
        b = b[:, 1]
        
        if np.remainder(b,xx) == 0:
            Loc[m + 1, 0] = b / xx
            Loc[m + 1, 1] = xx
        elif np.floor(b / xx) == 0:
            Loc[m + 1, 0] = 0
            Loc[m + 1, 1] = b
        else:
            Loc[m + 1, 0] = np.floor(b / xx)
            Loc[m + 1, 1] = b - xx * np.floor(b / xx)
        
        Sig[:, m + 1] = r[:, b].reshape(bnd)
        
    temp = np.copy(Loc[:, 0])
    Loc[:, 0] = Loc[:, 1]
    Loc[:, 1] = temp
    
    return Loc, Sig, xx

def PPI(imagecub, skewer_no):
    score = np.zeros([imagecub.shape[0] * imagecub.shape[1],1])
    
    skewer_sets = np.floor(skewer_no / 500.0) + 1
    last_skewer_no = np.mod(skewer_no, 500.0)
    
    for i in range(np.int(skewer_sets)):
        if skewer_sets - (i+1) == 0:
            skewer_no = last_skewer_no
        else:
            skewer_no = 500
            
        skewers = np.zeros([imagecub.shape[2], np.int(skewer_no)])
        
        for j in range(np.int(imagecub.shape[2])):
            for k in range(np.int(skewer_no)):
                skewers[j, k] = random.random() - 0.5
                
        for i in range(np.int(skewer_no)):
            skewers[:,i] = skewers[:, i] / np.linalg.norm(skewers[:, i])
            
        projcub = np.transpose(np.reshape(np.transpose(imagecub), [imagecub.shape[2], imagecub.shape[0] * imagecub.shape[1]]))
        proj_result = np.dot(projcub, skewers)
        
        for i in range(np.int(skewer_no)):
            max_pos = np.argwhere(proj_result[:, i] == np.max(proj_result[:, i]))
            min_pos = np.argwhere(proj_result[:, i] == np.min(proj_result[:, i]))
            score[max_pos] = score[max_pos] + 1
            score[min_pos] = score[min_pos] + 1
            
    result = np.argwhere(score > 0)
    result = result[:, 0]
    
    As = np.reshape(np.sort(np.transpose(score[:, 0])), [1, score.shape[0]])
    x = np.argwhere(np.diff(As)) + 1
    x = np.hstack([0, np.transpose(x[:, 1])])
    
    xx = np.zeros([1, x.shape[0]])
    
    for n in range(x.shape[0]):
        xx[0, n] = As[0, x[n]]
        
    y = np.argwhere(np.diff(As)) + 1
    
    yy = np.hstack([np.transpose(y[:, 1]), As.shape[1] + 1])
    
    y1 = np.hstack([0, np.transpose(y[:, 1]) + 1])
    
    yy = yy - y1
    
    index = np.zeros([1, np.max(yy) + 1])
    
    for q in range(index.shape[1]):
        index[0, q] = q
    
    return xx, yy, score, result

def SGA(HIM, p):
    n=1;
    z=0;

    endmember_index = np.array([[np.round(random.random() * HIM.shape[0])],[np.round(random.random() * HIM.shape[1])]])

    while n < p:
        endmember=[];
    
        for i in range(n):
            endmember.append(HIM[np.int(endmember_index[0,i]),np.int(endmember_index[1,i]),0:n])
    
        endmember = np.transpose(np.array(endmember))
        endmember = np.reshape(endmember,[endmember.shape[0],endmember.shape[1]])
        newendmember_index = np.zeros([2,1])
        maxvolume = 0
    
        for i in range(HIM.shape[0]):
            for j in range(HIM.shape[1]):
                a = HIM[i,j,0:n]
                a = np.reshape(a,[a.shape[0],1])
            
                jointpoint = np.hstack([endmember,a])
                
                jointmatrix = np.transpose(np.hstack([np.transpose(np.ones([1,n+1])),np.transpose(jointpoint)]))
            
                volume = abs(np.linalg.det(jointmatrix)) / math.factorial(n)
            
                if volume > maxvolume:
                    maxvolume = volume
                    newendmember_index[0,0] = i
                    newendmember_index[1,0] = j
                
        endmember_index = np.hstack([endmember_index,newendmember_index])
        
        n = n + 1
        if z == 0:
            n = 1
            endmember_index = np.reshape(endmember_index[:,1],[endmember_index.shape[0],1])
            z = z + 1
        
    endmember_index = endmember_index.transpose()
    
    return endmember_index

def NFINDR(HIM, p):
    endmemberindex = []
    
    newvolume = 0
    prevolume = -1
    
    row, column, band = HIM.shape
    
    switch_results = 1
    
    if band > p:
        use_svd = 1
    else:
        use_svd = 0
    
    for i in range(p):
        rand = np.random.rand()
        while 1:
            temp1 = np.round(row * rand)
            temp2 = np.round(column * rand)
            
            if (temp1 > 0 and temp2 > 0):
                break
            
        endmemberindex.append(np.hstack([temp1, temp2]))
        
    endmemberindex = (np.array(endmemberindex)).transpose().astype(np.int)
    
    print(endmemberindex)
    
    endmember = []
    
    for i in range(p):
        if use_svd:
            endmember.append(HIM[endmemberindex[0, i], endmemberindex[1, i], :].reshape(band))
    
    endmember = (np.array(endmember)).transpose()
    
    if use_svd:
        s = np.linalg.svd(endmember)[1]
        endmembervolume = 1
        
        for i in range(p):
            endmembervolume = endmembervolume * s[i]
            
    while newvolume > prevolume:
        for i in range(row):
            for j in range(column):
                for k in range(p):
                    caculate = endmember.copy()
                    
                    if use_svd:
                        caculate[:, k] = np.squeeze(HIM[i, j, :])
                        s = np.linalg.svd(caculate)[1]
                        volume = 1
                        
                        for z in range(p):
                            volume = volume * s[z]
                            
                    if volume > endmembervolume:
                        endmemberindex[:,k] = np.squeeze(np.vstack([i, j]))
                        endmember = caculate.copy()
                        endmembervolume = volume
                        
        prevolume = newvolume
        newvolume = endmembervolume
    
    if switch_results:
        endmemberindex = np.vstack([endmemberindex, endmemberindex[0, :].reshape(1, endmemberindex.shape[1])])
        endmemberindex = np.delete(endmemberindex, 0, 0)
        endmemberindex = endmemberindex.transpose()
        
    return endmemberindex