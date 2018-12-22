import numpy as np 
import cv2


img = cv2.imread("messi6.jpg",)
# print(cv2.IMREAD_COLOR,cv2.IMREAD_GRAYSCALE,cv2.IMREAD_UNCHANGED)
# b,g,r = cv2.split(img)
# img = cv2.merge([g,b,r])

cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("image",img)

k = cv2.waitKey(0)
# k = cv2.waitKey(0) & 0xFF
# if k == 27:
#     cv2.destroyAllWindows()
# elif k == ord("s"):
#     cv2.imwrite("messigray.png",img)
#     cv2.destroyAllWindows()
