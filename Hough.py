# coding=utf-8
import cv2
import numpy as np  
 
img = cv2.imread("test2.jpg")
img2gry = img.copy()
img2gry = cv2.cvtColor(img2gry,cv2.COLOR_BGR2GRAY)
img2gry = cv2.GaussianBlur(img2gry,(3,3),0)

# cv2.imshow('Result0', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(img[20][30])
edges = cv2.Canny(img, 40, 200, apertureSize = 3)
# cv2.imshow('Result1', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# lines = cv2.HoughLines(edges, 1.0, np.pi/180, 10)
# result = img.copy()
# for x1, y1, x2, y2 in lines[0]:
#     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
"""
    @.image： 必须是二值图像，推荐使用canny边缘检测的结果图像； 
    @.rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0 
    @.theta： 线段以弧度为单位的角度精度，推荐用numpy.pi/180 
    @.threshod: 累加平面的阈值参数，int类型，超过设定阈值才被检测出线段，
        值越大，基本上意味着检出的线段越长，检出的线段个数越少。根据情况推荐先用100试试
    @.lines：这个参数的意义未知，发现不同的lines对结果没影响，但是不要忽略了它的存在 
    @.minLineLength：线段以像素为单位的最小长度，根据应用场景设置 
    @.maxLineGap：同一方向上两条线段判定为一条线段的最大允许间隔（断裂），
        超过了设定值，则把两条线段当成一条线段，值越大，允许线段上的断裂越大，越有可能检出潜在的直线段
"""

# 经验参数
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1.0, np.pi/180, 100, minLineLength, maxLineGap)
print(len(lines))
lines = lines.reshape(-1,4)
for i in range(lines.shape[0]):
    x1,y1,x2,y2 = lines[i]
    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),2)

cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
