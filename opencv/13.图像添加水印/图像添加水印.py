import  cv2
import  numpy as np
image = cv2.imread('./lena.png')
image1 = cv2.imread('./logo.png')
row,col = image1.shape[:2]
roi = image[:row,:col]
image_gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY_INV)
image_and = cv2.bitwise_and(roi,roi,mask = mask)
dst = cv2.add(image_and,image1)
image[:row,:col] = dst
cv2.imshow('ronghe',image)
cv2.waitKey(0)