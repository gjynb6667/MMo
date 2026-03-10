import  cv2
import numpy as np
image = cv2.imread('./picture2.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, image_thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
contours,hierachy = cv2.findContours(image_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,contours,-1,(0,0,255),2)
for cnt in contours:
     rect = cv2.minAreaRect(cnt)
     box = np.int64(cv2.boxPoints(rect))
     cv2.drawContours(image,[box],-1,(255,0,0))
cv2.imshow('image',image)
cv2.waitKey(0)
