import  cv2
import  numpy as  np
image = cv2.imread('./picture5.png')
image_shape = image.shape
image_H = np.zeros(image_shape,dtype=np.uint8)
image_gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
image_canny = cv2.Canny(image_gray,30,70)
lines = cv2.HoughLinesP(image_canny,0.8,np.pi/180,90,minLineLength=50,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(image_H,(x1,y1),(x2,y2),(0,0,255))
cv2.imshow('image',image_H)
cv2.waitKey(0)