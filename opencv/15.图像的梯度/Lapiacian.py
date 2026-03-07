import cv2
image = cv2.imread('./shudu.png')
image_Lapiacian = cv2.Laplacian(image,-1)
cv2.imshow('La',image_Lapiacian)
cv2.waitKey(0)