import cv2
import numpy as np
from matplotlib import pyplot as plt
 


def mergedhistogram(mergedimage): 
    img1 = cv2.imread(mergedimage)

# Calculate histogram without mask
    hist1 = cv2.calcHist([img1],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([img1],[1],None,[256],[0,256])
    hist3 = cv2.calcHist([img1],[2],None,[256],[0,256])

#plt.subplot(111), plt.imshow(img1)
    plt.subplot(111), plt.plot(hist1), plt.plot(hist2),plt.plot(hist3),plt.title("Original Merged image before Encryption")
    plt.xlim([0,256])
    plt.show()


def encryptedhistogram(encryptedpath):
    
    img2 = cv2.imread(encryptedpath)
# Calculate histogram without mask
    hist1 = cv2.calcHist([img2],[0],None,[256],[0,256])
    hist2 = cv2.calcHist([img2],[1],None,[256],[0,256])
    hist3 = cv2.calcHist([img2],[2],None,[256],[0,256])

#plt.subplot(111), plt.imshow(img1)
    plt.subplot(111), plt.plot(hist1), plt.plot(hist2),plt.plot(hist3),plt.title("Encrypted image")
    plt.xlim([0,256])
    plt.show()
    

def dehistogram(demergedimage):
        img3 = cv2.imread(demergedimage)

# Calculate histogram without mask
        hist1 = cv2.calcHist([img3],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([img3],[1],None,[256],[0,256])
        hist3 = cv2.calcHist([img3],[2],None,[256],[0,256])

#plt.subplot(111), plt.imshow(img1)
        plt.subplot(111), plt.plot(hist1), plt.plot(hist2),plt.plot(hist3),plt.title("Merged Restored image after Decryption")
        plt.xlim([0,256])
        
        plt.show()