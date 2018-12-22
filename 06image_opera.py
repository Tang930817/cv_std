# >1. 2D卷积
import cv2
import numpy as np
from matplotlib import pyplot as plt
import skimage as si

img = cv2.imread('messi6.jpg',1)

# k_c = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(img,-1,k_c)
# dst = cv2.boxFilter(img,-1,(5,5))
# dst = cv2.GaussianBlur(img,(5,5),0)
B,G,R = cv2.split(img)
temp = cv2.merge([R,G,B])
# temp = img
noise = si.util.random_noise(temp,mode = 'gaussian')
# dst = cv2.bilateralFilter(temp,9,75,75)
# dst = cv2.medianBlur(noise)
k_e = np.ones((5,5),np.float32)
dst = cv2.erode(temp,k_e,iterations=1)

# cv2.imshow('iamge',img)
plt.subplot(131),plt.imshow(temp),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(132),plt.imshow(noise),plt.title('Noise')
plt.xticks([]),plt.yticks([])
plt.subplot(133),plt.imshow(dst),plt.title('filter')
plt.xticks([]),plt.yticks([])
plt.show()