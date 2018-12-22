import cv2
import numpy as np 


apple = cv2.imread('apple.jpg')
orange = cv2.imread('orange.jpg')

capple = apple.copy()
gua_apple_pyramid = [capple]
# 将apple进行高斯金字塔处理，总共6级处理，resol↓
for i in np.arange(6):
    capple = cv2.pyrDown(capple)
    gua_apple_pyramid.append(capple)

corange = orange.copy()
gua_orange_pyramid = [corange]
# 将orange进行高斯金字塔处理，总共6级处理，resol↓
for i in np.arange(6):
    corange = cv2.pyrDown(corange)
    gua_orange_pyramid.append(corange)

lup_apple_pyramid = [gua_apple_pyramid[5]]
# 将apple进行拉普拉斯金字塔处理，总共5级处理：高斯变换后的[第n层]—[(n-1)层升采样],差值存入列表
for i in np.arange(5,0,-1):
    gua_apple_E = cv2.pyrUp(gua_apple_pyramid[i])
    L_apple = cv2.subtract(gua_apple_pyramid[i-1],gua_apple_E)
    lup_apple_pyramid.append(L_apple)

lup_orange_pyramid = [gua_orange_pyramid[5]]
# 将orange进行拉普拉斯金字塔处理，总共5级处理，高斯变换后的[第n层]—[(n-1)层升采样],差值存入列表
for i in np.arange(5,0,-1):
    gua_apple_E = cv2.pyrUp(gua_orange_pyramid[i])
    L_orange = cv2.subtract(gua_orange_pyramid[i-1],gua_apple_E)
    lup_orange_pyramid.append(L_orange)

LS = [] # 存放拼接后的Lup图,将apple和orange的 （高斯N层-N-1层升采样）的差值  进行拼接，存入列表
        # 差值实际就是gauss升采样的处理结果
for la,lo in zip(lup_apple_pyramid,lup_orange_pyramid):
    rows,cols,dpt = la.shape
    # hstack 和 vstack分别：水平拼接和竖直拼接
    ls = np.vstack((la[:rows//2,:],lo[rows//2:,:]))   
    LS.append(ls)

ls_ = LS[0] # LS[0]是最小的一张图
for i in np.arange(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.vstack((apple[:rows//2,:],orange[rows//2:,:]))

# TODO test git
# cv2.imwrite('Pyramid_blending2.jpg',ls_)
# cv2.imwrite('Direct_blending.jpg',real)

# name_list=['0','1','2','3','4','5']
while True:
#     for i,item in zip(name_list,lup_apple_pyramid):
#         cv2.imshow('%s'%i,item)
    cv2.imshow('ls_',ls_)
    if cv2.waitKey(0) & 0xff == 27:
        break

cv2.destroyAllWindows()