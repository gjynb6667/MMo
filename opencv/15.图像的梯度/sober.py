import  cv2
image = cv2.imread('./shudu.png')
image_sober = cv2.Sobel(image,-1,1,0)
cv2.imshow('sober',image_sober)
cv2.waitKey(0)