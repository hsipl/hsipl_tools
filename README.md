## How to install 安裝方法  
```bash
pip install git+https://github.com/ek2061/hsipl_tools  
```

## python files 說明
1. anomaly_detection 異常檢測演算法  
2. band_selection 波段選擇演算法  
3. calc_matrix 計算相關矩陣、共變異矩陣  
4. endmember_extraction 端元選擇演算法  
5. feature_analysis 特徵分析演算法(HFC)  
6. model PCA和PLS  
7. preprocessing 預處理方法  
8. rpca_decomposition RPCA分解演算法  
9. target_detection 目標偵測演算法  
10. roi ROI方法

## Package Request 套件要求  
numpy  
matplotlib  
scipy  
pandas  
scikit-learn

## Sample Code 範例程式
```python
import scipy.io as sio  
img = sio.loadmat('panelHIM.mat')['HIM']  #讀圖檔  

from hsipl_tools import anomaly_detection as ad   
rxd_res = ad.R_rxd(img)  
``` 

## Code Contribution 程式碼貢獻
1. 學弟妹如果要貢獻新方法，創建dev分支再發PR給我，或是那個函式確定沒問題，直接推master也沒關係  
2. 在函式內稍微說明一下變數與函式用法即可，中英文皆可，格式任意，範例如下
```python
def print_num(a):
    '''
    印出輸入的數字
    
    param a: 任意數字, 型態為int 
    '''
    print(a)
```

## Wiki 維基
無聊可以去看也可以去編輯
