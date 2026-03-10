import cv2
import  numpy as  np
image = cv2.imread('./flower.png')
image_b = image + 200
cv2.imshow('imagh',image_b)
cv2.waitKey(0)