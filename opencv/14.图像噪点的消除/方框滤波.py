import  cv2
image  = cv2.imread('./lena.png')
image_box = cv2.boxFilter(image,-1,(3,3))
cv2.imshow('box',image_box)
cv2.waitKey(0)