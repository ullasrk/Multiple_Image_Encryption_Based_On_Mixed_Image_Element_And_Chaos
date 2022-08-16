import os
import glob
import cv2
import copysequencefilename as fcp
import image_merge as merge
import copysequencegenerater as cp
import easygui


def decryption(n,size):
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
        
    pwlcmimagepath=(r"C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\resized_images\\image1.jpg")
    initial=float(easygui.enterbox(msg='Enter Keys For Decryption \n \t Key 1.', title='Sequence Generator during Decryption', default='', strip=True))
    #initial=float(input("Enter the key 1 : "))
    p=float(easygui.enterbox(msg='Enter Keys For Decryption \n \t Key 2.', title='Sequence Generator during Decryption', default='', strip=True))
    #p=float(input("Enter the key 2 : "))
    i=1
    cp.pwlcm_values(pwlcmimagepath,initial,p,i)

    x=fcp.filename(n,i)
    lst=[]
    path=glob.glob("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\outputsegment\\*.jpg")
    i=0
    for file in path:
        y=os.path.basename(file)
        z=int(os.path.splitext(y)[0])
        lst.append(z)
        lst.sort()
    
    for i in range (n):
        if(lst[i]==x[i]):
            a=0
        else:
            a=1

    if(a==0):
        print("Match found Starting Decryption")
        merge.image_size(size,n)
    else:
        print("Keys Not Matched \nPlease Enter The keys Entered during Encryption")
    
    return a
    
    
    
