import  cv2
image = cv2.imread('./lena.png')
image_blur = cv2.blur(image,(3,3))
cv2.imshow('image_blur',image_blur)
cv2.waitKey(0)