# import cv2
# import numpy as np 
# from matplotlib import pyplot as plt 


"""
img1 = cv2.imread('messi6.jpg')
temp = img1

img2 = cv2.imread('logo.png')

rows,cols,channels = img2.shape
roi = img1[0:rows,0:cols]

print(img1[5][5])
"""


# # 掩膜
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret,mask = cv2.threshold(img2gray,175,255,cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
# img1_bg = cv2.bitwise_and(roi,roi,mask=mask)
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows,0:cols] = dst

# while True:
#     cv2.imshow('mask',mask)
#     cv2.imshow('mask_inv',mask_inv)
#     cv2.imshow('img1_bg',img1_bg)
#     cv2.imshow('img2_fg',img2_fg) 
#     cv2.imshow('dst',dst)
#     cv2.imshow('img1_rs',img1)         
#     if cv2.waitKey(0) & 0xff == 27:
#         break

# cv2.destroyAllWindows()
# # plt.subplot(221),plt.imshow(img1),plt.title('messi')
# # plt.subplot(222),plt.imshow(img2),plt.title('orange')
# # plt.subplot(223),plt.imshow(mask),plt.title('mask')
# # plt.subplot(224),plt.imshow(mask_inv),plt.title('mask_inv')
# # plt.show()


# green = np.array([0,255,0])
# hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
# print(hsv_green)

# img = cv2.imread('logo.png',0)
# print(img.shape)

# pic_list = []
# name_list = []
# img_ori = cv2.imread('apple.jpg')
# pic_list.append(img_ori)
# name_list.append('apple')
# img_conv = np.stack(img_ori,axis=2)
# pic_list.append(img_conv)
# name_list.append('apple_conv')

# while True:
#     for pic,name in zip(pic_list,name_list):
#         cv2.imshow(name, pic)
#     if cv2.waitKey(0) & 0xff == 27:
#         break
    
# cv2.destroyAllWindows()


# a = np.array([[1,2],
# [2,4]])
# b = np.array([
#     [1,1],
#     [1,1]
# ])
# c = np.subtract(b,a)
# print(c)
# ori = cv2.imread('apple.jpg')
# apple = cv2.imread('apple.jpg')
# for i in range(1):
#     apple = cv2.pyrDown(apple)
# for i in range(1):
#     apple = cv2.pyrUp(apple)
# D_value = np.subtract(ori,apple)

# while True:
#     cv2.imshow('apple',ori)
#     cv2.imshow('',apple)
#     cv2.imshow('dif',D_value)
#     if cv2.waitKey(0) & 0xff == 27:
#         break

# cv2.destroyAllWindows()

 
# img = cv2.imread('apple.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
# surf = cv2.xfeatures2d.SURF_create()
# kp = surf.detect(gray, None)
 
# img = cv2.drawKeypoints(gray, kp, img)
 
# cv2.imshow("img", img)
 
# k = cv2.waitKey(0)
# if k & 0xff == 27:
#     cv2.destroyAllWindows()

# #read image
# img = cv2.imread('messi6.jpg', cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_surf = img
# img_sift = img
# #SIFT
# surf = cv2.xfeatures2d.SURF_create()
# sift = cv2.xfeatures2d.SURF_create()
# keypoints_surf = surf.detect(gray,None)
# keypoints_sift = sift.detect(gray,None)
# #kp, des = surf.detectAndCompute(gray,None)  #des是描述子，for match， should use des, bf = cv2.BFMatcher();smatches = bf.knnMatch(des1,des2, k=2
# cv2.drawKeypoints(gray, keypoints_surf, img_surf)
# cv2.drawKeypoints(gray, keypoints_sift, img_sift)
# while True:
#     cv2.imshow('testSURF', img_surf)
#     cv2.imshow('testSift', img_sift)
#     if cv2.waitKey(0) & 0xff == 27:
#         break
# cv2.destroyAllWindows()

# help(surf.detect)



# -*- coding: utf-8 -*-
import cv2
import numpy as np
# from find_obj import filter_matches,explore_match
from matplotlib import pyplot as plt
 
# def getSift():
#     '''
#     得到并查看sift特征
#     '''
#     img_path1 = '../../data/home.jpg'
#     #读取图像
#     img = cv2.imread(img_path1)
#     #转换为灰度图
#     gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     #创建sift的类
#     sift = cv2.SIFT()
#     #在图像中找到关键点 也可以一步计算#kp, des = sift.detectAndCompute
#     kp = sift.detect(gray,None)
#     print type(kp),type(kp[0])
#     #Keypoint数据类型分析 http://www.cnblogs.com/cj695/p/4041399.html
#     print kp[0].pt
#     #计算每个点的sift
#     des = sift.compute(gray,kp)
#     print type(kp),type(des)
#     #des[0]为关键点的list，des[1]为特征向量的矩阵
#     print type(des[0]), type(des[1])
#     print des[0],des[1]
#     #可以看出共有885个sift特征，每个特征为128维
#     print des[1].shape
#     #在灰度图中画出这些点
#     img=cv2.drawKeypoints(gray,kp)
#     #cv2.imwrite('sift_keypoints.jpg',img)
#     plt.imshow(img),plt.show()
 
def matchSift():
    '''
    匹配sift特征
    '''
    img1 = cv2.imread('../../data/box.png', 0)  # queryImage
    img2 = cv2.imread('../../data/box_in_scene.png', 0)  # trainImage
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 蛮力匹配算法,有两个参数，距离度量(L2(default),L1)，是否交叉匹配(默认false)
    bf = cv2.BFMatcher()
    #返回k个最佳匹配
    matches = bf.knnMatch(des1, des2, k=2)
    # cv2.drawMatchesKnn expects list of lists as matches.
    #opencv2.4.13没有drawMatchesKnn函数，需要将opencv2.4.13\sources\samples\python2下的common.py和find_obj文件放入当前目录，并导入
    p1, p2, kp_pairs = filter_matches(kp1, kp2, matches)
    explore_match('find_obj', img1, img2, kp_pairs)  # cv2 shows image
    cv2.waitKey()
    cv2.destroyAllWindows()
 
def matchSift3():
    '''
    匹配sift特征
    '''
    img1 = cv2.imread('gui,jpg', 0)  # queryImage
    img2 = cv2.imread('gui2.jpg', 0)  # trainImage
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 蛮力匹配算法,有两个参数，距离度量(L2(default),L1)，是否交叉匹配(默认false)
    bf = cv2.BFMatcher()
    #返回k个最佳匹配
    matches = bf.knnMatch(des1, des2, k=2)
    # cv2.drawMatchesKnn expects list of lists as matches.
    #opencv3.0有drawMatchesKnn函数
    # Apply ratio test
    # 比值测试，首先获取与A 距离最近的点B（最近）和C（次近），只有当B/C
    # 小于阈值时（0.75）才被认为是匹配，因为假设匹配是一一对应的，真正的匹配的理想距离为0
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:10], None, flags=2)
    cv2.drawm
    plt.imshow(img3), plt.show()
 
matchSift3()
