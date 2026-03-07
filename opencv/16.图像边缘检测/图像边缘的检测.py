import  cv2
image =  cv2.imread('./picture.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_blur = cv2.GaussianBlur(image_gray,(5,5),1.5)
image_canny = cv2.Canny(image_blur,30,70)
cv2.imshow('canny',image_canny)
cv2.waitKey(0)