import  cv2
image = cv2.imread('./lena.png')
image_bil = cv2.bilateralFilter(image,5,151,151)
cv2.imshow('bil',image_bil)
cv2.waitKey(0)