# import cv2
# import numpy as np
# img=cv2.imread('messi6.jpg')
# px=img[100,100]
# print(px)
# blue=img[100,100,0]
# print(blue)
# print(img.shape)

import cv2
import numpy as np
# img=cv2.imread('messi6.jpg')
# head=img[190:250,210:285]
# img[60:120,210:285]=head
# img=cv2.imshow('test', img)
# cv2.waitKey(0)

img1 = cv2.imread('logo.png')
img2 = cv2.imread('messi6.jpg')

# img1 = img1[30:335,95:545]
# img2 = img2[30:335,30:480]

# dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
thresh_param, dst = cv2.threshold(img1,100,255,0)

cv2.imshow('', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
