import numpy as np
import cv2
import pandas as pd

def pwlcm(initial,p,r,c):
    t = initial
    res = []
    for i in range(r):
        res1=[]
        for j in range(c):
            if (t >= 0 and t < p):
                t = t/p
                res1.append(t)
            elif (t >= p and t<0.5):
                t = (t-p)/(0.5-p)
                res1.append(t)
            elif (t >= 0.5 and t<1):
                t = 1-t
                if (t >= 0 and t < p):
                    t = t/p
                    res1.append(t)
                elif (t >= p and t<0.5):
                    t = (t-p)/(0.5-p)
                    res1.append(t)
        res.append(res1)
    return res

def pwlcm_values(path,initial,p,i):
    img = cv2.imread(path)
    r, c, t = img.shape  

    value=pwlcm(initial,p,r,c)
    list.sort(value)
    keys=np.array(value)
    keys.reshape(r,c)
    #print(keys)
    power=(10**4)
    power_keys=np.multiply(keys,power)
    int_keys=power_keys.astype(int)
    #print("\n",int_keys)

    #demo=cv2.imread("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\image1.jpg")
    #print("original image values : \n",demo)

    my_df = pd.DataFrame(int_keys)
    my_df.to_csv('my_csv{}.csv'.format(i), index=False, header=False)

'''
encrypted_image=np.zeros((r,c))

print("encrypted image values before encryption; \n",encrypted_image)
'''
'''
#encrypted image
for height in range(r):
    for width in range(c):
        #encrypted_image[height,width]=demo[height, width]*int_keys[height, width]
        demo[height,width]=int_keys[height,width]
            
cv2.imshow("Encrypted image",demo)
print("image values after encryption : \n",demo)
cv2.imwrite("C:\\Users\\Dell\\OneDrive\\Desktop\\pwlcmencrypt\\encrypted_image.png",demo)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
decrypted_image = np.zeros((r, c, t), np.uint8)
for row in range(r):
    for column in range(c):
        for depth in range(t):
            decrypted_image[row, column, depth] = encrypted_image[row, column, depth] ^ key[row, column, depth] # C ^ B = A, C XOR B = A
            
cv2.imshow("Decrypted Image", decrypted_image)
cv2.imwrite('C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\decrypted_image.png',decrypted_image)

cv2.waitKey()
cv2.destroyAllWindows()
'''