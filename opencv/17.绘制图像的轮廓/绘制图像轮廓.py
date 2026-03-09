import  cv2
image = cv2.imread('./picture2.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret, image_thrash = cv2.threshold(image_gray,127,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
contours,hierarchy = cv2.findContours(image_thrash,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,contours,-1,(0,0,255))
cv2.imshow('contours',image)
cv2.waitKey(0)