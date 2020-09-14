# hsipl_tools
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
