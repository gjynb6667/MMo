import cv2
#
# if __name__ == "__main__":
#     path = "./lena.png"
#     image_np = cv2.imread(path)
#     image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)  # 灰度化
#     image_np_adaptive = cv2.adaptiveThreshold(image_np_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 10)  # 自适应二值化
#     cv2.imshow("image_np_gray", image_np_gray)
#     cv2.imshow("image_np_adaptive", image_np_adaptive)
#     cv2.waitKey(0)
image = cv2.imread('./lena.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_thresh = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blockSize=3,C=10)
cv2.imshow('gray',image_gray)
cv2.imshow('thresh',image_thresh)
cv2.waitKey(0)
