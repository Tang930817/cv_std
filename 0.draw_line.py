import cv2
import numpy as np 


img = np.zeros((480,640,3),dtype=np.uint8)
cv2.line(img,(0,0),(480,640),(0,0,255),5)

cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imshow("image",img)
cv2.waitKey(0)

cv2.destroyAllWindows()