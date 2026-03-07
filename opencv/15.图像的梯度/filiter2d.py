import cv2
import  numpy as  np
image = cv2.imread('./shudu.png')
kernel = np.array([[-1,0,1],
                   [-2,0,1],
                   [-1,0,1]])
image_filter = cv2.filter2D(image,-1,kernel)
cv2.imshow('filter',image_filter)
cv2.waitKey(0)
