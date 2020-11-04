# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 20:15:04 2020

@author: Yuchi
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

class model:
    def PCA(self, data, n_components):
        '''
        Principal component analysis (PCA).
        
        param data: data, shape is [x, y, bands] or [x, bands].
        param n_components: Number of components to keep.
        '''
        if data.ndim > 2:
            data = data.reshape(-1, data.shape[2])
        pca_model = PCA(n_components)
        pca_res = pca_model.fit_transform(data)
        return pca_model, pca_res
    
    def PLS(self, data, ans, n_components):
        '''
        PLS regression
        
        param data: data, shape is [x, y, bands] or [x, bands].
        param ans: one hot ans, shape is [data num, kinds num].
        param n_components: Number of components to keep.
        '''
        if data.ndim > 2:
            data = data.reshape(-1, data.shape[2])
        pls = PLSRegression(n_components)  
        pls_model = pls.fit(data, ans)
        return pls_model
    
    def PCA_PLS(self, HIM, kinds, points, pca_components = 4, pls_components = 3, show_band = 150, isPaint = True):
        from matplotlib import pyplot as plt
        
        plt.figure()
        plt.imshow(HIM[:, :, show_band])
        
        # 選點訓練
        xy=np.array(plt.ginput(kinds*points), 'i')
        d = []
        for x, y in xy:
            d.append(HIM[y, x])
        d = np.array(d)
        
        # 3類各三個點做 onehot encode        
        y = np.array([[i]*points for i in range(kinds)]).reshape(-1)
        n_values = np.max(y) + 1
        y_onehot = np.eye(n_values)[y]
        plt.close()
        
        # d做pca拿到模型與降維後的結果
        pca_model, d_pca_res = self.PCA(d, pca_components)
        
        # img做pca拿到降維後的結果
        img_pca = pca_model.transform(HIM.reshape(-1, HIM.shape[2]))
    
        # d做pls拿到model
        pls_model = self.PLS(d_pca_res, y_onehot, pls_components)
        
        # 預測
        img_predict = pls_model.predict(img_pca.reshape(-1, pca_components))
        img_result = np.argmax(img_predict, -1).reshape(HIM.shape[0], HIM.shape[1])
        
        if isPaint:
            plt.figure()
            plt.imshow(img_result)
        
        return img_result