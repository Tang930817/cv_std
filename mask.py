import cv2
import numpy as np 

apple_img = cv2.imread('apple.jpg')
mask = np.zeros(apple_img.shape[:2],np.uint8)

mask[100:200,100:250] = 255
masked_apple = cv2.bitwise_and(apple_img,apple_img,mask = mask)

cv2.imshow('mask',masked_apple)
cv2.waitKey(0)
cv2.destroyAllWindows()