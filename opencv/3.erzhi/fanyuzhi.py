import cv2
import numpy as np

image = cv2.imread('./flower.png')
image_gary = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
thresh = 127
maxval = 255 
image_shape = image_gary.shape
image_trash = np.zeros((image_shape[0],image_shape[1]),dtype=np.uint8)
# for i in range(image_shape[0]):
#     for j in range(image_shape[1]):
#         if image_gary[i,j] > thresh:
#             image_trash[i,j] = 0
#         else:
#             image_trash[i,j] = maxval
# cv2.imshow('gray',image_gary)
# cv2.imshow('erzhi',image_trash)
# cv2.waitKey(0)
ret, image_thresh =   cv2.threshold(image_gary,thresh,maxval,cv2.THRESH_BINARY_INV)
cv2.imshow('thresh',image_thresh)
cv2.waitKey(0)