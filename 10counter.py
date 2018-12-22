import cv2
import numpy as np 


img = cv2.imread('messi6.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)
edge = cv2.Canny(img_blur, 50, 175)

c_edge = edge.copy()
result,contours,hierarcy = cv2.findContours(c_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
cv2.drawContours(img, contours, -1, (0, 255, 255), 3)
# print(contours,hierarcy)

cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()