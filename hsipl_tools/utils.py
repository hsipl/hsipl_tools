# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:53:27 2021

@author: Yuchi
"""

'''
這個py檔存著Yuchi寫的沒用小工具
'''

import numpy as np
import matplotlib.pyplot as plt

def get_thres_n_times(img, n):
    '''
    給圖片和Otsu次數，返回threshold
    '''
    from skimage.filters import threshold_otsu
        
    img = img.reshape(-1)
    for i in range(n-1):
        thresh = threshold_otsu(img)
        bimage = img.copy()
		
        bimage[bimage < thresh] = 0
        bimage[bimage >= thresh] = 1
        
        bimage = bimage.reshape((bimage.shape[0] * bimage.shape[1], 1))

        zero_index = np.argwhere(bimage == 0)
        zero_index = list(zero_index[:, 0])
        
        img = np.delete(img, zero_index, 0)
    
    threshold = threshold_otsu(img)
        
    return threshold

def thresholding(img, threshold):
    '''
    給圖片和threshold 返回二值化圖片
    '''
    img_cp = img.copy()
    img_cp[img >= threshold] = 1
    img_cp[img < threshold] = 0
    
    return img_cp

def plot_confusion_matrix(y_true, y_pred, classes,  # classes請給標籤種類array
                          normalize=False,  # 有正規化會是小數(比例)
                          title=None, cmap=plt.cm.Blues,  # 色系詳細參照以下網址
                          # https://matplotlib.org/examples/color/colormaps_reference.html
                          true_label='True label', predicted_label='Predicted label'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.utils.multiclass import unique_labels
    from sklearn.metrics import confusion_matrix

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    classes = np.array(classes)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel=true_label,
           xlabel=predicted_label)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax


def 存彩色圖(圖片, 檔名='res'):
    '''
    給彩色圖和檔名，存一張沒有邊界留白和刻度的圖片
    '''
    height, width, _ = 圖片.shape
    fig = plt.figure('res')
    plt.axis('off')
    fig.set_size_inches(width/100.0, height/100.0)  # 輸出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(圖片)
    plt.show()
    plt.savefig(f'{檔名}.jpeg', pad_inches=0.0)
    
def 存灰階圖(圖片, 檔名='res'):
    '''
    給灰階圖和檔名，存一張沒有邊界留白和刻度的圖片
    '''
    height, width = 圖片.shape
    fig = plt.figure('res')
    plt.axis('off')
    fig.set_size_inches(width/100.0, height/100.0)  # 輸出width*height像素
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.imshow(圖片, 'gray')
    plt.show()
    plt.savefig(f'{檔名}.jpeg', pad_inches=0.0)