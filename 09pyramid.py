import cv2
import numpy as np 


LeftImg = cv2.imread('apple.jpg')
RightImg = cv2.imread('orange.jpg')

cLeftImg = LeftImg.copy()
gua_LeftImg_pyramid = [cLeftImg]
# 将LeftImg进行高斯金字塔处理，总共6级处理，resol↓
for i in np.arange(6):
    cLeftImg = cv2.pyrDown(cLeftImg)
    gua_LeftImg_pyramid.append(cLeftImg)

cRightImg = RightImg.copy()
gua_RightImg_pyramid = [cRightImg]
# 将RightImg进行高斯金字塔处理，总共6级处理，resol↓
for i in np.arange(6):
    cRightImg = cv2.pyrDown(cRightImg)
    gua_RightImg_pyramid.append(cRightImg)

lup_LeftImg_pyramid = [gua_LeftImg_pyramid[5]]
# 将LeftImg进行拉普拉斯金字塔处理，总共5级处理：高斯变换后的[第n层]—[(n-1)层升采样],差值存入列表
for i in np.arange(5,0,-1):
    gua_LeftImg_E = cv2.pyrUp(gua_LeftImg_pyramid[i])
    L_LeftImg = cv2.subtract(gua_LeftImg_pyramid[i-1],gua_LeftImg_E)
    lup_LeftImg_pyramid.append(L_LeftImg)

lup_RightImg_pyramid = [gua_RightImg_pyramid[5]]
# 将RightImg进行拉普拉斯金字塔处理，总共5级处理，高斯变换后的[第n层]—[(n-1)层升采样],差值存入列表
for i in np.arange(5,0,-1):
    gua_LeftImg_E = cv2.pyrUp(gua_RightImg_pyramid[i])
    L_RightImg = cv2.subtract(gua_RightImg_pyramid[i-1],gua_LeftImg_E)
    lup_RightImg_pyramid.append(L_RightImg)

LS = [] # 存放拼接后的Lup图,将LeftImg和RightImg的 （高斯N层-N-1层升采样）的差值  进行拼接，存入列表
        # 差值实际就是gauss升采样的处理结果
for la,lo in zip(lup_LeftImg_pyramid,lup_RightImg_pyramid):
    rows,cols,dpt = la.shape
    # hstack 和 vstack分别：水平拼接和竖直拼接
    # ls = np.vstack((la[:rows//2,:],lo[rows//2:,:]))
    ls = np.hstack((la[:,:int(0.5*cols)],lo[:,int(0.5*cols):]))    
    LS.append(ls)

ls_ = LS[0] # LS[0]是最小的一张图
for i in np.arange(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
# real = np.vstack((LeftImg[:rows//2,:],RightImg[rows//2:,:]))
real = np.hstack((LeftImg[:,:int(0.5*cols)],RightImg[:,int(0.5*cols):]))

cv2.imwrite('Pyramid_blending2.jpg',ls_)
cv2.imwrite('Direct_blending.jpg',real)
# cv2.imshow('Pyramid_blending2.jpg',ls_)
# cv2.imshow('Direct_blending.jpg',real)

cv2.waitKey(0)
cv2.destroyAllWindows()