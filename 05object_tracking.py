# import cv2
# import numpy as np

# cap = cv2.VideoCapture(r'vedio.mp4')

# while True:

#     # Take each frame
#     _, frame = cap.read()

#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # define range of blue color in HSV
#     lower_blue = np.array([110,50,50])
#     upper_blue = np.array([130,255,255])

#     # Threshold the HSV image to get only blue colors
#     masks = cv2.inRange(hsv, lower_blue, upper_blue)

#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(frame,frame, mask= masks)

#     cv2.imshow('frame',frame)
#     cv2.imshow('mask',masks)
#     cv2.imshow('res',res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# img=cv2.imread('messi5.jpg')
# # 下面的 None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子
# # 因此这里为 None
# # res=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# #OR
# # 这里呢，我们直接设置输出图像的尺寸，所以不用设置缩放因子
# height,width=img.shape[:2]
# # print(img.shape[0],img.shape[1],img.shape[2])
# # print(height,width)
# res=cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)
# while(1):
#     cv2.imshow('res',res)
#     cv2.imshow('img',img)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()
# # Resize(src, dst, interpolation=CV_INTER_LINEAR)

# import cv2
# import numpy as np

# img = cv2.imread('messi5.jpg',0)
# rows,cols = img.shape

# M = np.float32([[1,0,50],[0,1,50]])
# dst = cv2.warpAffine(img,M,(cols,rows))

# cv2.imshow('img',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# img=cv2.imread('messi6.jpg',0)
# rows,cols=img.shape
# # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
# # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
# M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
# # 第三个参数是输出图像的尺寸中心
# dst=cv2.warpAffine(img,M,(2*cols,2*rows))
# while(1):
#     cv2.imshow('img',dst)
#     if cv2.waitKey(1)&0xFF==27:
#         break
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread('messi6.jpg',0)
# ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])

# plt.show()

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img = cv2.imread('messi6.jpg',0)
# img = cv2.medianBlur(img,5)

# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,11,2)

# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]

# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()