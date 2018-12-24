import cv2
import numpy as np
import time
img = cv2.imread('logo.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
edge = cv2.Canny(img_blur, 70, 200)
print(edge)
c_edge = edge.copy()


result,contours,hierarcy = cv2.findContours(c_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
"""
@.mode 表示轮廓的检索模式
    1. cv2.RETR_LIST  只是提取所有的轮廓，而不去创建任何父子关系
    2. cv2.RETR_EXTERNAL  只会返回最外边的的轮廓，所有的子轮廓都会被忽略掉
    3. RETR_CCOMP  会返回所有的轮廓并将轮廓分为两级组织结构
    4. RETR_TREE  会返回所有轮廓，并且创建一个完整的组织结构列表
@.method 轮廓的近似办法
    1.  cv2.CHAIN_APPROX_NONE  存储所有的轮廓点，相邻的两个点的像素位置差不超过1,
        即max(abs(x1-x2),abs(y2-y1)) <= 1
    2.  cv2.CHAIN_APPROX_SIMPLE  压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标,
        例如一个矩形轮廓只需4个点来保存轮廓信息
    3.  cv2.CHAIN_APPROX_TC89_L1 & cv2.CHAIN_APPROX_TC89_KCOS
        使用teh-Chinl chain 近似算法
"""

# Image Moments
# print(cv2.moments(contours[3]))

# Contours Area
# print(cv2.contourArea(contours[3]))

# Boundingrect
x, y, w, h = cv2.boundingRect(contours[3])
# img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)

# minAreaRect
rect = cv2.minAreaRect(contours[3])
print(rect)
box = cv2.boxPoints(rect)
print(box)
box = np.int64(box)
print(box)
img = cv2.drawContours(img, [box], -1, (0, 0, 255), 2)

# contours drawing
# cv2.drawContours(img, contours, -1, (0, 255, 255), 2)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# histogram 统计直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
print(hist)