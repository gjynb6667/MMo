import cv2
import  numpy as  np
image1 = cv2.imread('./picture4.png')
image2 = cv2.imread('./muban.png')
image1_gray = cv2.cvtColor(image1,cv2.COLOR_BGRA2GRAY)
image2_gray = cv2.cvtColor(image2,cv2.COLOR_BGRA2GRAY)
h,w = image2.shape[0:2]
res = cv2.matchTemplate(image1_gray,image2_gray,cv2.TM_CCOEFF_NORMED)
threshold = 0.57
location = np.where(res > threshold)
for left_top in zip(*location[::-1]):
    right_bottom =  (left_top[0]+w,left_top[1]+h)
    cv2.rectangle(image1,left_top,right_bottom,(0,0,255))
cv2.imshow('imgae',image1)
cv2.waitKey(0)