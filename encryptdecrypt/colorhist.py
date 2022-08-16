import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pandas as pd


def colorhist(path,j,name):
   
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
    plt.subplots_adjust(hspace=1)
    k=name
    fig.suptitle("{} Image {}".format(k,j) ,fontweight="bold",fontsize=18, y=1.1)
    fig.text(0.5, 0.0005, 'Range Of Pixels', ha='center')
    fig.text(0.07, 0.5, '%of Pixels', va='center', rotation='vertical')
    lim=255
    img = cv.imread(path)
    color = ('Blue','Green','Red')
    
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        histr /= histr.sum()
        axs[i].plot(histr,color = col)
        axs[i].title.set_text(col)
        plt.xlim([0,256])
    plt.show()

#colorhist("C:\\Users\\ABHIRAM\\Desktop\\encryptdecrypt\\Merged_image.jpg", 0,"k")