import numpy as np
from panorama import Stitcher
import cv2 as cv
import imutils

img1 = cv.imread('0.png')
img2 = cv.imread('1.png')
#img1 = imutils.resize(img1, width=400)
#img2 = imutils.resize(img2, width=400)

stitcher = Stitcher()
(result, vis) = stitcher.stitch([img1, img2], showMatches=True)
# (result, vis) = stitch([img1,img2], showMatches=True)


cv.imshow('Result', result)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('gg_con.jpg',result)