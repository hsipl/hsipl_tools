import numpy as np
from skimage import segmentation, measure, filters

def dust_roi(img, mask,ignoreArea=10):
    '''
    param img: signal, type is 3d-array,size is [x,y,band num]    
    param mask: signal, type is 2d-array, size is [x,y]
    ignoreArea:signal,type is int ,default 10
    '''
    try:
        label_roi = []
        cleared = mask.copy()
        segmentation.clear_border(cleared)
        label_image = measure.label(cleared)
        borders = np.logical_xor(mask, cleared)
        label_image[borders] = -1
        x = 5
        for region in measure.regionprops(label_image):
            # 忽略小區域
            if region.area < ignoreArea:
                continue
            # ROI
            minr, minc, maxr, maxc = region.bbox
            label_roi.append(img[minr-x:maxr+x, minc-x:maxc+x, :])
    except Exception as e:
        print(e)
    else:
        return label_roi