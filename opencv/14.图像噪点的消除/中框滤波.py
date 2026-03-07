import  cv2
image = cv2.imread('./lena.png')
image_median = cv2.medianBlur(image,3)
cv2.imshow('median',image_median)
cv2.waitKey(0)