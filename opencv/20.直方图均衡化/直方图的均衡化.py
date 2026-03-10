import  cv2
import numpy as np

image = cv2.imread('./picture3.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
def calcAndDrawHist(image_gray):
    hist = cv2.calcHist([image_gray] , [0], None, [256], [0, 256])
    minval,maxval,minloc,maxloc = cv2.minMaxLoc(hist)
    histimg = np.zeros((256,256,3),dtype=np.uint8)
    temp = int(256*0.9)
    for h in range(256):
        intensity = int(temp*hist[h]/maxval)
        cv2.line(histimg,(h,256),(h,256-intensity),(255,0,0))
    return histimg
image_his = calcAndDrawHist(image_gray)
clahe = cv2.createCLAHE(2,(8,8))
image_clahe = clahe.apply(image_gray)

image_equalizeHist = cv2.equalizeHist(image_gray)
image_equalizeHist_image = calcAndDrawHist(image_equalizeHist)
cv2.imshow('his',image_his)
cv2.waitKey(0)