## target_detection
cem, subset_cem, sw_cem, hcem  
sam_img, sam_point, ed_img, ed_point, sid_img, sid_point  
sid_tan, sid_sin, rsdpw_sam, rsdpw_sid, rsdpw_sid_tan, rsdpw_sid_sin
ace, mf  
kmd_img, kmd_point, rmd_img, rmd_point
kmfd, rmfd, K_rxd, R_rxd  
lptd, utd, rxd_utd  
calc_R, calc_K_u  

## Package Request:  
numpy  
matplotlib  
scipy  
pandas  

## 懶得包起來了，git clone到site-packages就可以直接import了
![image](https://github.com/ek2061/hsipl_tools/blob/master/cmd.png)

## 範例程式
import scipy.io as sio  
img = sio.loadmat('panelHIM.mat')['HIM']  #讀圖檔  

from hsipl_tools import target_detection as td   
rxd_res = td.R_rxd(img)  
