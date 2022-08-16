import os
import cv2
import numpy as np
from numpy import random
from PIL import Image
import math
import sys
import glob 
import colorhist as histr1
import image_merge as merge
import copysequencegenerater as cp
import imagesegment as segment
import decryption as decrypt
import imagecorelation as corelation
import npcr as np_cr
import uaci as unified
import easygui
import decrypt_segment as dec

def calcEntropy(img):
    entropy = []

    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en
    
    

n =int(easygui.enterbox(msg='Enter No Of Images To Encrypt.', title='Image Encryption ', default='', strip=True))
size =int(easygui.enterbox(msg='Enter image size.', title=' Image Encryption ', default='', strip=True))
initial=float(easygui.enterbox(msg='Enter Keys For Encryption \n \t Key 1.', title='Key Generator', default='', strip=True))
p=float(easygui.enterbox(msg='Enter Keys For Encryption \n \t Key 2.', title='Key Generator', default='', strip=True))


path=glob.glob("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\exp\\*.jpg")
i=1
for file in path:
    print(file)
    dp_image = cv2.imread(file)
    resized=cv2.resize(dp_image,(size, size))
    cv2.imshow("Image {}".format(i),resized)
    cv2.imwrite("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\resized_images\\image"+str(i)+".jpg", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if(i==n):
        break
    i+=1

random.seed(12345)    #random number seed, this can be a "locked" seed - for frequency analysis testing or we can use a TRNG to make fully random. If no seed is used, the PRNG uses the systems hardware as an entropy source, using the systems clock for example.

merge.image_size(size,n)


# Load original image
demo = cv2.imread("Merged_image.jpg")
r, c, t = demo.shape

# Display original image
cv2.imshow("Merged image", demo)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Create random key
key = random.randint(256, size = (r, c, t))

# Encryption
# Iterate over the image
encrypted_image = np.zeros((r, c, t), np.uint8)
for row in range(r):
    for column in range(c):
        for depth in range(t):
            encrypted_image[row, column, depth] = demo[row, column, depth] ^ key[row, column, depth]   
            
cv2.imshow("Encrypted image", encrypted_image)
cv2.imwrite('C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\encrypted_image.jpg',encrypted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


############################################################################################################
pwlcmimagepath=(r"C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\resized_images\\image1.jpg")
i=0
cp.pwlcm_values(pwlcmimagepath,initial,p,i)

dir = 'C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\outputsegment'
if os.listdir(dir)!= []:
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))


segment.image_segment_values(size,n,i)

#print("Encrypted images Segmented based on entered Key values")
easygui.msgbox("Encrypted images Segmented based on entered Key values will be displayed", title="simple gui")
path=glob.glob("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\outputsegment\\*.jpg")
i=1
for file in path:
    print(file)
    dp_image = cv2.imread(file)
    x=os.path.basename(file)
    cv2.imshow("Image {}".format(x),dp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if(i==n):
        break
    i+=1

#user_ans=input("To Start Decryption Press Y or N to Stop at the encryption : ")
user_ans=easygui.ynbox('Continue with Decryption Process', 'Title', ('Yes', 'No'))
if(user_ans == 1):
    a=decrypt.decryption(n,size)
else:
    sys.exit(0)

# Decryption
# Iterate over the encrypted image
if(a==0):
    easygui.msgbox("Keys entered matched Starting Decryption Process", title="simple gui")
    decrypted_image = np.zeros((r, c, t), np.uint8)
    for row in range(r):
        for column in range(c):
            for depth in range(t):
                decrypted_image[row, column, depth] =encrypted_image[row, column, depth] ^ key[row, column, depth] # C ^ B = A, C XOR B = A
            
    cv2.imshow("Decrypted Image", decrypted_image)
    cv2.imwrite('C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\decrypted_image.png',decrypted_image)
            
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    easygui.msgbox("Keys entered during Encryption Does Not Match Stopping the Process", title="simple gui")
    sys.exit(0)

easygui.msgbox("Decrypted segmented original images are displayed",title="simple gui")
i=1
dir = 'C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\decrypted'
if os.listdir(dir)!= []:
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
        
        
    
dec.image_segment_values(size,n,i)


path=glob.glob("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\decrypted\\*.jpg")
i=1
for file in path:
    print(file)
    dp_image = cv2.imread(file)
    x=os.path.basename(file)
    cv2.imshow("Image {}".format(x),dp_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if(i==n):
        break
    i+=1

################################################################################################### 
# Histogram Analysis

for i in range(1,n+1):
    path=(r"C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\exp\\{}.jpg".format(i))
    histr1.colorhist(path,i,"Original")

mergepath='C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\Merged_image.jpg'
demergepath='C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\decrypted_image.png'
encryptedpath='C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\encrypted_image.jpg'

#x=histr.mergedhistogram(mergepath)
x1=histr1.colorhist(mergepath,0,"Merged")
#y=histr.encryptedhistogram(encryptedpath)
y1=histr1.colorhist(encryptedpath,0,"Encrypted")
z1=histr1.colorhist(demergepath,0," Merged Decrypted")

for i in range(1,n+1):
    path=(r"C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\exp\\{}.jpg".format(i))
    histr1.colorhist(path,i,"Decrypted")

#####################################################################################################
#finding snr value

original_image=Image.open("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\Merged_image.jpg")
image_snr=np.array(original_image)
mean_image=np.mean(image_snr)

img1=Image.open("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\decrypted_image.png")
noisey_image=np.array(img1)
noise=noisey_image-image_snr
mean_noise=np.mean(noise)
noise_diff=noise-mean_noise
var_noise=np.sum(np.mean(noise_diff**2))
if(var_noise==0):
    snr=100
else:
    snr=(np.log10(mean_image/var_noise))*20

print("snr value of image is :", snr)
easygui.msgbox("\n SNR Value = {}".format(snr), title="SNR Value")

#####################################################################################################
#finding psnr value

mse=float(np.mean((demo-decrypted_image)**2))
if(mse==0):
    print("PSNR Value : 0")
    easygui.msgbox("PSNR Value = {}".format(mse), title="PSNR Value")
else:
    max_pixel=255.0
    psnr=20*math.log10(max_pixel/math.sqrt(mse))
    easygui.msgbox("PSNR Value = {}".format(psnr), title="PSNR Value")
    print("PSNR Value is",psnr)
    
#####################################################################################################
#Information Entropy analysis

Entropy_img1 = cv2.imread("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\Merged_image.jpg", cv2.IMREAD_GRAYSCALE)
Entropy_img2 = cv2.imread("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\decrypted_image.png", cv2.IMREAD_GRAYSCALE)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
entropy1 = calcEntropy(Entropy_img1)
entropy2 = calcEntropy(Entropy_img2)

easygui.msgbox("Entropy Of Merged Image = {} \nEntropy Of Decrypted Image = {}".format(entropy1,entropy2), title="Entropy Of Merged Image")
#easygui.msgbox("Entropy Of Decrypted Image = {}".format(entropy2), title="Entropy Of Decrypted Image ")

print("Entropy value of image before encryption :",entropy1)
print("Entropy value of image after  encryption :",entropy2)


#####################################################################################################
#finding corelation 

loc1=(r"C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\Merged_image.jpg")
loc2=(r"C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\encrypted_image.jpg")
'''
cor_value=corelation.correlation(loc1,loc2)
print("\nCorrelation value of Two images is : \n \n",cor_value)
easygui.msgbox("Correlation value of Two images  \n \n {}".format(cor_value), title="CORRELATION VALUE")
'''
##########################################################################################################
# finding variance
'''
var_value=corelation.variance(loc2)
print("\n Variance of Image : \n \n ",var_value)
easygui.msgbox("Variance Value = {}".format(var_value), title="Variance")
'''
##########################################################################################################
# finding npcr

npcr_value=np_cr.npcrv(loc1,loc2)
print("\n Number of pixel changed rate : %.2f" %npcr_value)

easygui.msgbox("Number of pixel changed rate = {0:.2f}".format(npcr_value), title="Number of pixel changed rate (NPCR)")

###########################################################################################################
# finding UACI 

uaci_value=unified.uaci(loc1,loc2)
print("\n Unified average avereage inensity changed rate : %.2f" %uaci_value)

easygui.msgbox("Unified average avereage inensity changed rate = {0:.3f}".format(uaci_value), title="Unified average inensity changed rate (UACI)")

###########################################################################################################
#finding rootmeansquare
'''
rmse=unified.rootmeansquareerror(loc1,loc2)
print("\n Root mean sqaure error of images is : %.3f" %rmse)

easygui.msgbox("Root mean sqaure error of images is = {0:.3f}".format(rmse), title="Root Mean square error of merged and decrypted images")
'''