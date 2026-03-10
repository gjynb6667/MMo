import cv2
import  numpy as np
image = cv2.imread('./picture5.png')
image_shape = image.shape
image_circle = np.zeros(image_shape,dtype=np.uint8)
image_gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
circles = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT,1,20,param1=70,param2=50)
circles = np.intp(np.around(circles))
print(circles)
for circle in circles:
    x,y,radius =circle[0]
    cv2.circle(image_circle,(x,y),radius,(0,0,255))
cv2.imshow('image',image_circle)
cv2.waitKey(0)