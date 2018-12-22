import cv2
import numpy as np 
from matplotlib import pyplot as plt 


img = cv2.imread("messi5.jpg")
lower_reso = cv2.pyrDown(img)
high_reso = cv2.pyrUp(lower_reso)

print(img.shape)
print(lower_reso.shape)
print(high_reso.shape)
# while True:
#     cv2.imshow("image",img)
#     if cv2.waitKey(0) & 0xff ==27:
#         break
# cv2.destroyAllWindows()
plt.subplot(131),plt.imshow(img),plt.title("Orig")
plt.xticks(),plt.yticks()
plt.subplot(132),plt.imshow(lower_reso),plt.title("Low_Resol")
plt.subplot(133),plt.imshow(high_reso),plt.title("High_Resol")
# plt.xticks([]),plt.yticks([])
plt.show()