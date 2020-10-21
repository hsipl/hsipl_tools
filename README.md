## target_detection 目標偵測演算法  
cem, subset_cem, sw_cem, hcem  
sam_img, sam_point, ed_img, ed_point, sid_img, sid_point  
sid_tan, sid_sin, rsdpw_sam, rsdpw_sid, rsdpw_sid_tan, rsdpw_sid_sin  
ace, mf  
kmd_img, kmd_point, rmd_img, rmd_point
kmfd, rmfd, K_rxd, R_rxd, sw_R_rxd  
lptd, utd, rxd_utd  
tcimf, cbd_img, cbd_point, td_img, td_point  
calc_R, calc_K_u  
LSOSP, KLSOSP  
NWHFC, HFC,  
AMSD  

## band_selection 波段選擇演算法  
CEM_BCC, CEM_BCM, CEM_BDM  
BS_STD, BS_Corrcoef, BS_Entropy  
BS_minV_BP, BS_maxV_BP, BS_SF_CTBS, BS_SB_CTBS  
minV_BP, maxV_BP, SF_CTBS, SB_CTBS, uniform_BS  
FminV_BP, BmaxV_BP, SF_TCIMBS, SB_TCIMBS  

## preprocessing 預處理  
data_normalize, msc, snv  
awgn  

## endmember_extraction 端元選擇演算法  
ATGP, PPI, SGA, NFINDR  

## Package Request 套件要求  
numpy  
matplotlib  
scipy  
pandas  

## How to use 使用方法  
懶得包起來了，在site-packages的位置git clone就可以直接import  
### Anaconda3
C:\Users\user\Anaconda3\Lib\site-packages
### 原生python
C:\Users\user\AppData\Roaming\Python\Python38\site-packages

git clone https://github.com/ek2061/hsipl_tools.git

## Sample 範例程式
import scipy.io as sio  
img = sio.loadmat('panelHIM.mat')['HIM']  #讀圖檔  

from hsipl_tools import target_detection as td   
rxd_res = td.R_rxd(img)  
