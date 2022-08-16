import math
import numpy as np
import random
import pandas as pd
#import copysequencegenerater as cp                                                                                                                            

def filename(no_of_images,i):
    slist=[]
    fslist=[]
    b=[]
    n=no_of_images

    file=open("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\my_csv{}.csv".format(i))
    a=np.loadtxt(file, delimiter=",")
    b=a.flatten()
    
    for i in range (n):
        yi_dash=math.floor((b[i])*(10**2))
        slist.append(yi_dash)


    for i in range(n):
        y_dash=slist[i]*n
        fslist.append(y_dash)
        
    return fslist
