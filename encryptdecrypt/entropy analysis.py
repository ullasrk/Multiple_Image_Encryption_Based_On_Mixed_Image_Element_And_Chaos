import cv2
import numpy as np


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


if __name__ == '__main__':
    Entropy_img1 = cv2.imread("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\Merged_image.jpg", cv2.IMREAD_GRAYSCALE)
    Entropy_img2 = cv2.imread("C:\\Users\\Dell\\OneDrive\\Desktop\\encryptdecrypt\\decrypted_image.png", cv2.IMREAD_GRAYSCALE)

    entropy1 = calcEntropy(Entropy_img1)
    entropy2 = calcEntropy(Entropy_img2)

    print("Entropy value of image before encryption :",entropy1)
    print("Entropy value of image after  encryption :",entropy2)