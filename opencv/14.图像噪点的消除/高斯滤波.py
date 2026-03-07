import  cv2
image = cv2.imread('./lena.png')
image_Gaussian = cv2.GaussianBlur(image,(3,3),sigmaX=1)
cv2.imshow('Gaussion',image_Gaussian)
cv2.waitKey(0)