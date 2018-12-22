import numpy as np 
import cv2

cap = cv2.VideoCapture("vedio.mp4")
width, height = cap.get(3), cap.get(4)
print(width,height)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 0.5)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 0.5)
flag = 1
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray)
    # print(ret)
    print(frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # print(cap.get(0))


cap.release()
cv2.destroyAllWindows()