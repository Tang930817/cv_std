# import cv2
# # events=[i for i in dir(cv2) if 'EVENT'in i]
# # print(events) 
# # events = []
# # for i in dir(cv2):
# #     if "EVENT" in i:
# #         events.append(i)
# img = cv2.imread("messi5.jpg")
# # print(events)

"""
def event_t(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("ok")
while True:
    cv2.imshow("image",img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
    cv2.setMouseCallback("image",event_t)
cv2.destroyAllWindows()
"""
import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()
