## anomaly_detection 異常檢測演算法  

## band_selection 波段選擇演算法  

## calc_matrix 計算相關矩陣、共變異矩陣  

## endmember_extraction 端元選擇演算法  

## feature_analysis 特徵分析演算法(HFC)  

## model 懶還要更懶函數(PCA和PLS)  

## preprocessing 預處理方法  

## rpca_decomposition RPCA分解演算法  

## target_detection 目標偵測演算法  

## Package Request 套件要求  
numpy  
matplotlib  
scipy  
pandas  
scikit-learn

## How to install 安裝方法  
pip install git+https://github.com/ek2061/hsipl_tools  

## Sample 範例程式
import scipy.io as sio  
img = sio.loadmat('panelHIM.mat')['HIM']  #讀圖檔  

from hsipl_tools import target_detection as td   
rxd_res = td.R_rxd(img)  
