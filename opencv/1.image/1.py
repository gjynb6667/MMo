import  numpy as np
import  cv2
import  matplotlib.pyplot as  plt
image = np.zeros((700,700,3),dtype=np.uint8)

black_size = 100
for i in range(0,700,black_size):
    for j in range(0,700,black_size):
        top_left = (j, i)
        bottom_right = (j + black_size -1, i + black_size -1)
        # image[i , :, :] = (255,255,255)
        # image[: , j, :] = (255,255,255)
        # if i != 0 and j != 0 and i != 600 and j != 600 and (i == j or i + j == 600):
        #     image[i:i + black_size, j:j + black_size, :] = [0, 0, 255]
        if i != 0 and i !=600 and ( i == j or i + j == 600):
            cv2.rectangle(image,top_left,bottom_right,(0,0,255),-1)
        else:
            cv2.rectangle(image,top_left,bottom_right,(255,255,255),1)
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()
#用切片方法创建三通道
b = image[:,:,0]
g = image[:,:,1]
r = image[:,:,2]

cv2.imshow("b",b)
cv2.imshow("g",g)
cv2.imshow("r",r)
#第二种:使用cv2.split()函数去分割
#b,g,r = cv2.split(image)
#创建新的图像，用来展示三通道图
blue_channel = np.zeros((700,700,3),dtype=np.uint8)
green_channel = np.zeros((700,700,3),dtype=np.uint8)
red_channel = np.zeros((700,700,3),dtype=np.uint8)
blue_channel[:,:,0] = b
red_channel[:,:,2] = r
green_channel[:,:,1] = g
blue_channel_rgb = cv2.cvtColor(blue_channel,cv2.COLOR_BGR2RGB)
green_channel_rgb = cv2.cvtColor(green_channel,cv2.COLOR_BGR2RGB)
red_channel_rgb = cv2.cvtColor(red_channel,cv2.COLOR_BGR2RGB)
plt.subplot(131)
plt.imshow(blue_channel_rgb)
plt.title('Blue_channel')
plt.axis('off')
plt.subplot(132)
plt.imshow(green_channel_rgb)
plt.title('Green_channel')
plt.axis('off')
plt.subplot(133)
plt.imshow(red_channel_rgb)
plt.title('Red_channel')
plt.axis('off')
plt.show()
# cv2.imshow('image',image)
# cv2.waitKey(0)
