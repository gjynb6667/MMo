import  cv2
image = cv2.imread('./picture1.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
ret,image_thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)
contours,hierachy = cv2.findContours(image_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
hull = cv2.convexHull(cnt)
image_poly = cv2.polylines(image,[hull],True,(0,0,255))
cv2.imshow('poly',image_poly)
cv2.waitKey(0)