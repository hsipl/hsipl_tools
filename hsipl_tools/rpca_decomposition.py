# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 14:32:48 2020

@author: user
"""

'''
RPCA分解法
程式是Wen Long學長寫的，註解是Yuchi寫的ㄏ

param data: 高光譜影像2D

return L, S: 都是2D的，要畫圖要自己轉回3D

example:
    >> x, y, z = HIM.shape
    >> r = np.reshape(HIM, [-1, z])
    >> L, S = hsipl_tools.rpca_decomposition.GA(r)
'''

import numpy as np
import numpy.matlib as mb
from scipy.stats import trim_mean
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def GA(data):
    X = data.transpose()

    K = 1
    epsilon = 10 * np.finfo(float).eps
    
    N, D = X.shape
    
    vectors = np.zeros([D, K])
    
    vectors[:] = np.NAN
    
    for k in range(K):
        mu = np.random.rand(D, 1) - 0.5
        
        mu = mu / np.linalg.norm(mu)
        
        for iterate in range(3):
            dots = np.dot(X, mu)
            mu = (np.dot(dots.transpose(), X)).transpose()
            mu = mu / np.linalg.norm(mu)
            
        for iterate in range(N):
            prev_mu = mu.copy()
            dot_signs = np.sign(np.dot(X, mu))
            mu = np.dot(dot_signs.transpose(), X)
            mu = (mu / np.linalg.norm(mu)).transpose()
            
            if np.max(abs(mu - prev_mu)) < epsilon:
                break
            
        if k == 0:
            vectors[:, k] = mu.reshape(D)
            X = X - np.dot(np.dot(X, mu), mu.transpose())
            
    new_min = np.min(data[:])
    new_max = np.max(data[:])
    
    L = nma_rescale(vectors, new_min, new_max)
    
    L = mb.repmat(L, 1, data.shape[1])
    
    S = data - L
    
    return L, S

def GM(data):
    X = data.transpose()
    
    K = 1
    epsilon = 1e-5
    
    N, D = X.shape
    
    vectors = np.zeros([D, K])
    
    vectors[:] = np.NAN
    
    for k in range(K):
        mu = np.random.rand(D, 1) - 0.5
        
        mu = mu / np.linalg.norm(mu)
        
        for iterate in range(3):
            dots = np.dot(X, mu)
            mu = (np.dot(dots.transpose(), X)).transpose()
            mu = mu / np.linalg.norm(mu)
            
        for iterate in range(N):
            prev_mu = mu.copy()
            dot_signs = np.sign(np.dot(X, mu))
            mu = (np.median(X / dot_signs, 0)).reshape(D, 1)
            mu = mu[:] / np.linalg.norm(mu)
            
            if np.max(abs(mu - prev_mu)) < epsilon:
                break
            
        if k == 0:
            vectors[:, k] = mu.reshape(D)
            X = X - np.dot(np.dot(X, mu), mu.transpose())
            
    new_min = np.min(data[:])
    new_max = np.max(data[:])
    
    L = nma_rescale(vectors, new_min, new_max)
    
    L = mb.repmat(L, 1, data.shape[1])
    
    S = data - L
    
    return L, S

def Godec(data):
    x, y = data.shape
    
    rank = 1
    card = x * y
    power = 0
    iter_max = 1e+2
    error_bound = 1e-3
    iterate = 1
    
    RMSE = []
    
    if x < y:
        data = data.transpose()
        
    L = data.copy()
    S = csr_matrix(np.zeros([x, y])).toarray()
    
    while True:
        Y2 = np.random.randn(y, rank)
        
        for i in range(power + 1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.transpose(), Y1)
            
        Q, R = np.linalg.qr(Y2)
        
        L_new = np.dot(np.dot(L, Q), Q.transpose())
        
        T = L - L_new + S
        
        L = L_new.copy()
        
        idx = (np.argsort(-1 * abs(T.reshape(1, x * y)))).reshape(x * y)
        
        S = np.zeros([x * y])
        
        S[idx[0:card]] = T.reshape(x * y)[idx[0:card]]
        
        S = S.reshape(x, y)
        
        T.reshape(x * y)[idx[0:card]] = 0
        
        RMSE.append(np.linalg.norm(T.reshape(x * y)))
        
        if (RMSE[-1] < error_bound) or (iterate > iter_max):
            break
        else:
            L = L + T
            
        iterate = iterate + 1
        
    LS = L+S
    
    error = np.linalg.norm(LS.reshape(x * y) - data.reshape(x * y)) / np.linalg.norm(data.reshape(x * y))
    
    if x < y:
        LS = LS.transpose()
        L = L.transpose()
        S = S.transpose()
        
    return L, S

def GreGoDec(M):
    rank = 1
    tau = 7
    power = 5
    tol = 1e-3
    k = 1
    
    m, n = M.shape
    
    if m < n:
        M = M.transpose()
        
    normD = np.linalg.norm(M[:])
    
    rankk = np.int(np.round(rank / k))
    
    error = np.zeros([rankk*power, 1])
    
    X, s, Y = svds(M, k, which='LM')
    
    X = X * s
    
    L = np.dot(X, Y)
    
    S = wthresh(M - L, 's', tau)
    
    T = M - L - S
    
    error = []
    
    error.append(np.linalg.norm(T[:]) / normD)
    
    iii = 1
    alf = 0
    stop = False
    
    for r in range(rankk):
        alf = 0
        increment = 1
        
        for iterate in range(power):
            X = np.dot(L, Y.transpose())
            
            X, R = np.linalg.qr(X)
            
            Y = np.dot(X.transpose(), L)
            
            L = np.dot(X, Y)
            
            T = M - L
            
            S = wthresh(T, 's', tau)
            
            T = T - S
            
            ii = iii + iterate
            
            error.append(np.linalg.norm(T[:]) / normD)
            
            if error[ii] < tol:
                stop = True
                break
                
            ratio = error[ii] / error[ii-1]
            
            if ratio >= 1.1:
                increment = np.max([0.1 * alf, 0.1 * increment])
                X = X1.copy()
                Y = Y1.copy()
                S = S1.copy()
                T = T1.copy()
                error[ii] = error[ii-1]
                alf = 0
            elif ratio > 0.7:
                increment = np.max([increment, 0.25 * alf])
                alf = alf + increment
                
            X1 = X.copy()
            Y1 = Y.copy()
            L1 = L.copy()
            S1 = S.copy()
            T1 = T.copy()
            
            L = L + (1 + alf) * T
            
            if stop:
                break
            
    L = np.dot(X, Y)
    
    if m < n:
        L = L.transpose()
        S = S.transpose()
        
    S = M - L
    
    return L, S

def OPRMF(data):
    X = normalize(data)
    
    rk = 2
    lambdaU = 1
    lambdaV = 1
    tol = 1e-2
    
    x, y = X.shape
    
    mask = np.ones([x, y])
    
    maxIter = 40
    startIndex = 1
    
    U = np.random.randn(x, rk)
    
    V = np.random.randn(rk, startIndex)
    
    lambd = 1
    eps = 1e-3
    
    IS = csr_matrix(np.eye(rk)).toarray()
    
    A = []
    B = []
    
    TA = []
    TB = []
    
    forgetFactor = 0.98
    confidence = 1e-3
    
    L = np.zeros([x, y])
    
    for j in range(y):
        Y = X[:, 0:j+1]
        
        if j != 0:
           V = np.hstack([V, V[:, j-1].reshape(V.shape[0], 1)])
           
        r = abs(Y - np.dot(U, V))
        
        confidence = np.min(confidence * 1.5)
        
        c = 0
        
        while True:
            c = c + 1
            
            oldR = r.copy()
            
            r = abs(Y - np.dot(U, V))
            
            r = (r < eps).astype(np.int) * eps + (r > eps).astype(np.int) * r
            
            r = np.sqrt(lambd) / r
            
            if j == (startIndex - 1):
                s = 0
            else:
                s = j
                
            for i in range(s, j+1, 1):
                temp1 = csr_matrix(r[:, i].reshape(x, 1)).toarray() * mask[:, i].reshape(x, 1)
                
                T = (U * temp1).transpose()
                        
                V[:, i] = (np.dot(np.linalg.inv(np.dot(T, U) + lambdaV * IS), np.dot(T, (Y[:, i]).reshape(x, 1)))).reshape(T.shape[0])
                
            r = abs(Y - np.dot(U, V))
            
            r = r.transpose()
            
            r = (r < eps).astype(np.int) * eps + (r > eps).astype(np.int) * r
            
            r = confidence * np.sqrt(lambd) / r
            
            if j == (startIndex - 1):
                A = []
                B = []
                for i in range(x):
                    T = np.dot(V, (np.diag(csr_matrix(r[:, i].reshape(r.shape[0], 1)).toarray() * mask[i, 0:startIndex])).reshape(r.shape[0], 1))
                
                    A.append(np.linalg.inv(np.dot(T, V.transpose()) + lambdaU * IS))
                    B.append(T * Y[i, :].reshape(Y.shape[1], 1))
                    U[i, :] = (np.dot(A[i], B[i])).reshape(1, T.shape[0])
            else:
                v = V[:, j].reshape(V.shape[0], 1)
                
                TA = A
                TB = B
                
                for i in range(x):
                    temp = np.dot(A[i], v) / forgetFactor
                    
                    if mask[i, j] == 0:
                        U[i, :] = (np.dot(TA[i], TB[i])).reshape(1, temp.shape[0])
                        continue
                    else:
                        TA[i] = A[i] / forgetFactor - r[j, i] * np.dot(temp, temp.transpose()) / (1 + r[j, i] * np.dot(v.transpose(), temp))
                        TB[i] = B[i] * forgetFactor + r[j, i] * Y[i, j] * v
                    
                    U[i, :] = (np.dot(TA[i], TB[i])).reshape(1, temp.shape[0])
                    
            r = abs(Y - np.dot(U, V))
            
            if j == (startIndex - 1):
                if ((np.sum(abs(r[:] - oldR[:]), 0) / np.sum(oldR[:])) < tol) and (c != 1) or (c > maxIter):
                    L[:, j] = (np.dot(U, V)).reshape(x)
                    break
            elif ((np.sum(abs(r[:, j] - oldR[:, j]), 0) / np.sum(oldR[:, j], 0)) < tol) or (c > maxIter):
                A = TA
                B = TB
                L[:, j] = (np.dot(U, V[:, j].reshape(V.shape[0], 1))).reshape(x)
                break
        
    S = X - L
        
    return L, S

def PCP(M):
    lambd = 1 / np.sqrt(np.max(M.shape))
    tol = 1e-5
    beta = 0.25 / np.mean(abs(M[:]))
    maxit = 1000
    
    m, n = M.shape
    
    S = np.zeros([m, n])
    
    L = np.zeros([m, n])
    
    Lambd = np.zeros([m, n])
    
    for i in range(maxit):
        nrmLS = np.linalg.norm(np.hstack([S, L]))
        
        X = Lambd / beta + M
        
        Y = X - L
        
        dS = S.copy()
        
        S = np.sign(Y) * (abs(Y) - lambd / beta)
        
        dS = S - dS
        
        Y = X - S
        
        dL = L
        
        U, D, V = svdecon(Y)
        
        VT = V.transpose()
        
        D = np.diag(D)
        
        ind = np.argwhere(D > (1 / beta))
        
        D = np.diag(D[ind] - (1 / beta))
        
        L = np.dot(np.dot(U[:, ind[0].reshape(1)], D.reshape(1, 1)), VT[ind[0].reshape(1), :])
        
        dL = L - dL
        
        RelChg = np.linalg.norm(np.hstack([dS, dL])) / (1 + nrmLS)
        
        if RelChg < tol:
            break
        
        Lambd = Lambd - beta * (S + L - M)
        
    return L, S

def PRMF(data):
    X = normalize(data)
    
    rk = 2
    lambdaU = 1
    lambdaV = 1
    tol = 1e-2
    maxIter = 40
    
    m, n = X.shape
    
    U = np.random.randn(m, rk)
    
    V = np.random.randn(rk, n)
    
    lambd = 1
    eps = 1e-3
    
    r = abs(X - np.dot(U, V))
    
    r = (r < eps).astype(np.int) * eps + (r > eps).astype(np.int) * r
    
    r = lambd / r
    
    c = 0
    
    IS = csr_matrix(np.eye(rk)).toarray()
    
    while True:
        c = c + 1
        oldR = r.copy()
        
        for i in range(n):
            temp1 = csr_matrix(r[:, i].reshape(m, 1)).toarray()
            
            T = (U * temp1).transpose()
                    
            V[:, i] = (np.dot(np.linalg.inv(np.dot(T, U) + lambdaV * IS), np.dot(T, (X[:, i]).reshape(m, 1)))).reshape(T.shape[0])
            
            r[:, i] = (abs(X[:, i].reshape(m, 1) - np.dot(U, V[:, i].reshape(V.shape[0], 1)))).reshape(m)
            
            r[:, i] = (r[:, i] < eps).astype(np.int) * eps + (r[:, i] > eps).astype(np.int) * r[:, i]
            
            r[:, i] = lambd / r[:, i]
        
        for i in range(m):
            T = np.dot(V, np.diag((csr_matrix(r[i, :].reshape(n, 1)).toarray()).reshape(n)))
            
            U[i, :] = (np.dot(np.linalg.inv(np.dot(T, V.transpose()) + lambdaU * IS), np.dot(T, (X[i, :]).reshape(n, 1)))).reshape(T.shape[0])
            
            r[i, :] = (abs(X[i, :].reshape(n, 1) - np.dot(V.transpose(), U[i, :].reshape(U.shape[1], 1)))).reshape(n)
            
            r[i, :] = (r[i, :] < eps).astype(np.int) * eps + (r[i, :] > eps).astype(np.int) * r[i, :]
            
            r[i, :] = lambd / r[i, :]
        
        if ((np.sum(abs(r.reshape(m * n) - oldR.reshape(m * n)), 0) / np.sum(oldR[:].reshape(m * n), 0)) < tol) and (c != 1) or (c > maxIter):
            break
    
    L = np.dot(U, V)
    
    S = X - L
    
    return L, S

def SSGoDec(M):
    rank = 1
    tau = 8
    power = 0
    iter_max = 1e+2
    error_bound = 1e-3
    iterate = 1
    
    RMSE=[]
    
    m, n = M.shape
    
    if m < n:
        M = M.transpose()
        
    L = M.copy()
    
    S = csr_matrix(np.zeros([m, n])).toarray()
    
    while True:
        Y2 = np.random.randn(n, rank)
        
        for i in range(power + 1):
            Y1 = np.dot(L, Y2)
            Y2 = np.dot(L.transpose(), Y1)
            
        Q, R = np.linalg.qr(Y2)
            
        L_new = np.dot(np.dot(L, Q), Q.transpose())
        
        T = L - L_new + S
            
        L = L_new.copy()
        
        S = wthresh(T, 's', tau)
        
        T = T - S
        
        RMSE.append(np.linalg.norm(T[:]))
        
        if (RMSE[-1] < error_bound) or (iterate > iter_max):
            break
        else:
            L = L + T
            
        iterate = iterate + 1
        
    LS = L+S
    
    error = np.linalg.norm(LS[:] - M[:]) / np.linalg.norm(M[:])
    
    if m < n:
        LS = LS.transpose()
        L = L.transpose()
        S = S.transpose()
        
    S = M - L
    
    return L, S

def SVT(D):
    lambd = 1 / np.sqrt(np.max(D.shape))
    tau = 1e+4
    delta = 0.9
    EPSILON_PRIMAL = 5e-4
    MAX_ITER = 25000
    
    m, n = D.shape
    
    Y = np.zeros([m, n])
    
    A = np.zeros([m, n])
    
    E = np.zeros([m, n])
    
    iterate = 0
    
    converged = False
    
    while not(converged):
        iterate = iterate + 1
        
        U, S, V = np.linalg.svd(Y, 0)
        
        A = np.dot(np.dot(U, np.diag(pos(S - tau))), V)
        
        E = np.sign(Y) * pos(abs(Y) - lambd * tau)
        
        M = D - A - E
        
        Y = Y + delta * M
        
        if ((np.linalg.norm(D - A - E, 'fro') / np.linalg.norm(D, 'fro')) < EPSILON_PRIMAL) or (iterate >= MAX_ITER):
            converged = True
            
    L = A.copy()
    S = E.copy()
            
    return L, S

def TGA(data):
    X = data.transpose()
    
    percent = 0.5
    K = 1
    
    N, D = X.shape
    
    vectors = np.zeros([D, K])
        
    vectors[:] = np.NAN
    
    epsilon = 1e-5
    
    for k in range(K):
        mu = np.random.rand(D, 1) - 0.5
            
        mu = mu / np.linalg.norm(mu)
        
        for iterate in range(3):
            dots = np.dot(X, mu)
            mu = (np.dot(dots.transpose(), X)).transpose()
            mu = mu / np.linalg.norm(mu)
            
        for iterate in range(N):
            prev_mu = mu.copy()
            dot_signs = np.sign(np.dot(X, mu))
            mu = trim_mean(X * dot_signs.reshape(N, 1), percent)
            mu = (mu[:] / np.linalg.norm(mu)).reshape(D, 1)
            
            if np.max(abs(mu - prev_mu)) < epsilon:
                break
                
        if k == 0:
            vectors[:, k] = mu.reshape(D)
            X = X - np.dot(np.dot(X, mu), mu.transpose())
            
    new_min = np.min(data[:])
    new_max = np.max(data[:])
    
    L = nma_rescale(vectors, new_min, new_max)
    
    L = mb.repmat(L, 1, data.shape[1])
    
    S = data - L
            
    return L, S

def nma_rescale(A, new_min, new_max):
    current_max = np.max(A[:])
    current_min = np.min(A[:])
    C =((A - current_min) * (new_max - new_min)) / (current_max - current_min) + new_min
    
    return C

def normalize(X):
    m, n = X.shape
    
    X = X - np.dot(np.ones([m, 1]), (np.mean(X, 0)).reshape(1, n))
    
    DTD = np.dot(X.transpose(), X)
    
    invTrX = np.ones([n, 1]) / (np.sqrt(np.diag(DTD))).reshape(n, 1)
    
    mul = np.dot(np.ones([m, 1]), invTrX.transpose())
    
    X = X * mul
    
    return X

def pos(A):
    return A * np.double(A > 0)

def svdecon(X):
    C = np.dot(X.transpose(), X)
    
    D, V = np.linalg.eig(C)
    
    ix = np.argsort(-1 * abs(D))
    
    V = V[:, ix]
    
    U = np.dot(X, V)
    
    s = np.sqrt(D[ix])
    
    U = U / s.reshape(1, s.shape[0])
    
    S = np.diag(s)
    
    return U, S, V

def wthresh(X, SORH, T):
    if (SORH == 'h'):
        Y = X * (np.abs(X) > T)
        return Y
    elif (SORH == 's'):
        res = (np.abs(X) - T)
        res = (res + np.abs(res))/2.
        Y = np.sign(X) * res
        return Y