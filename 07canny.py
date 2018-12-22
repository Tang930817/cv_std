import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import time

img = cv2.imread("messi6.jpg")
img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# B,G,R = cv2.split(img)
# temp = cv2.merge([R,G,B])

filt = cv2.GaussianBlur(img,(5,5),0)

dst = cv2.Canny(filt,100,200)
# while True:
# #     time.sleep(2)
#     cv2.imshow('gray',img2gray)
#     time.sleep(2)
#     break
plt.subplot(131),plt.imshow(img2gray),plt.title("Original(gray)")
plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(filt),plt.title("Gauss——filter")
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(dst),plt.title("Canny")
plt.xticks([]),plt.yticks([])
plt.show()

cv2.bitwise_and(img,roi,)
# plt.subplot(131),plt.imshow(temp),plt.title('Original')
# plt.xticks([]),plt.yticks([])