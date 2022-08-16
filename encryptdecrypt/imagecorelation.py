import matplotlib
import numpy as np
import cv2
from PIL import Image, ImageStat
from scipy import ndimage
import numpy as np

def correlation(loc1,loc2):
    im1=cv2.imread(loc1)
    im2=cv2.imread(loc2)
    a1=np.asarray(im1)
    print(a1)
    a2=np.asarray(im2)
    print("\n \n",a2)
    cm = np.corrcoef(a1.flat,a2.flat)
    
    return cm

def variance(loc):
    im = Image.open(loc)
    stat = ImageStat.Stat(im)
    
    return(stat.var)
