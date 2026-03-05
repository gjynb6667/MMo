import cv2
import numpy as np

image = cv2.imread('./flower.png')
image_gary = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

min_value = image_gary.min()
max_value = image_gary.max()
image_shape = image_gary.shape
image_thresh = np.zeros((image_shape[0],image_shape[1]),dtype=np.uint8)
thresh = 127
maxval = 255
n_0 =0
n_1 =0
w_0 =0
w_1 =0
u_0 =0
u_1 =0
u =0
rows = image_shape[0]
cols = image_shape[1]

var = {}
for t in range(min_value+1,max_value,1):
    forground = []
    background = []
    forepix = 0
    backpix = 0
    pix = 0
    for i in  range(image_shape[0]):
        for j in range(image_shape[1]):
            if image_gary[i,j] > t:
                forground.append(image_gary[i,j])
                forepix += image_gary[i,j]
                pix += image_gary[i,j]
            else:
                background.append(image_gary[i,j])
                backpix += image_gary[i,j]
                pix += image_gary[i,j]
    n_0 = len(forground)
    n_1 = len(background)
    w_0 = n_0 / image_shape[0] * image_shape[1]
    w_1 = n_1 / image_shape[0] * image_shape[1]
    u_0 = forepix / n_0
    u_1 = backpix / n_1
    u = pix / image_shape[0] * image_shape[1]
    g = w_0 * ((u_1 - u) ** 2) + w_1 * ((u_0 - u) ** 2)
    var[t] = g
thresh = max(var,key= var.get)
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        if image_gary[i,j] > thresh:
            image_thresh[i,j] = maxval
        else:
            image_thresh[i,j] = 0
cv2.imshow("thresh",image_thresh)
cv2.imwrite('./test.png',image_thresh)
cv2.waitKey(0)



