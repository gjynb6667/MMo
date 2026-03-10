import cv2
image = cv2.imread('./picture2.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
ret,image_thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
contours,hierarchy = cv2.findContours(image_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,contours,-1,(0,255,255),2)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    top_left = (x,y)
    bottom_right = (x+w,y+h)
    cv2.rectangle(image,top_left,bottom_right,(0,0,255),2 )
cv2.imshow('image',image)
cv2.waitKey(0)