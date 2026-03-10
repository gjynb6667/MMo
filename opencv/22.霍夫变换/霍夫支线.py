import  cv2
import numpy as np
image = cv2.imread('./picture5.png')
image_shape = image.shape
image_Houghlines = np.zeros(image_shape,dtype=np.uint8)
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_canny = cv2.Canny(image_gray,30,70)
lines = cv2.HoughLines(image_canny,0.8,np.pi/180,90)
for line in lines:
    rho,theta = line[0]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x1,x2 = 0,image_shape[1]
    y1 = int((rho - x1*cos_theta)/sin_theta)
    y2 = int((rho - x2*cos_theta)/sin_theta)
    cv2.line(image_Houghlines,(x1,y1),(x2,y2),(0,0,255))
cv2.imshow('imahge',image_Houghlines)
cv2.waitKey(0)