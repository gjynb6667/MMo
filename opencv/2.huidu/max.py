import  cv2
import numpy as np
image = cv2.imread('./flower.png')
print(image)
image_shape  = image.shape
image_gray = np.zeros((image_shape[0],image_shape[1]),dtype= np.uint8)
weight_red = 0.299
weight_green = 0.587
weight_blue = 0.114
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        image_gray[i][j] = max(image[i][j][0]   ,image[i][j][1]  ,image[i][j][2] )
cv2.imshow('image_gray',image_gray)
cv2.waitKey(0)
