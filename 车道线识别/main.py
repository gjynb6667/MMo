import  hqyj_mqtt
import  queue
import  base64
import  numpy as  np
import cv2
def base_64_up(image):
    #b54decode解码
    a = base64.b64decode(image['image'])
    #frombuffer转为np类型
    b = np.frombuffer(a,dtype= np.uint8)
    #imdecode图像解码
    c = cv2.imdecode(b,cv2.IMREAD_COLOR)
def change_eye(image):
    image_shape = image.shape
    image_size = (image_shape[1],image_shape[0])
    #设置图像截图点坐标
    src = np.float32([[80,image_shape[0]],
    [int(image_shape[1]/2-60),int(image_shape[0]/2)],
    [int(image_shape[1]/2+60),int(image_shape[0]/2)],
    [450,image_shape[0]]])
    #设置图像显示点坐标
    drt = np.float32([[image_shape[1]/4,image_shape[0]],
           [image_shape[1]/4, 0],
           [image_shape[1]*3/4, 0],
           [image_shape[1]*3/4, image_shape[0]]])
    #设置变换矩阵,输入原坐标点和要变换后的坐标点
    M = cv2.getPerspectiveTransform(src,drt)
    #进行图像校正输入原图像，矩阵，图像尺寸，变换模式
    image_warp = cv2.warpPerspective(image,M,image_size,flags=cv2.INTER_LINEAR)
    return  image_warp
def get_line_tidu(image):
    #先高斯滤波去除噪点
    img_Gauss = cv2.GaussianBlur(image,(5,5),sigmaX=1)
    #灰度化
    image_gray = cv2.cvtColor(img_Gauss,cv2.COLOR_BGRA2GRAY)
    #用sober算法得到梯度图
    ret = cv2.Sobel(image_gray,-1,1,0)
    #去除模糊和噪点
    ret, image_thresh = cv2.threshold(ret,127,255,cv2.THRESH_BINARY)
    return image_thresh
#膨胀和腐蚀让车道线更明显
def dilate_erode(image,kenerl_size):
    kenel = np.ones((kenerl_size,kenerl_size),np.uint8)
    ret = cv2.dilate(image,kenel,iterations=1)
    image_erode = cv2.erode(ret,kenel,iterations=1)
    return  image_erode
#hls比hsv抗干扰能力强
def hlsSelect(img,thresh = (220,255)):
    #转换为hls模型
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    #提取出l通道
    l_channel = hls[:,:,1]
    #把l通道里的像素值都改为0~255的范围
    l_channel = l_channel/np.max(l_channel)*255
    #创建一个全黑的图像
    binary = np.zeros_like(l_channel)
    #把在(220,255)范围内的改为白色
    binary[(l_channel>thresh[0])&(l_channel<thresh[1])] = 1
    return binary
#运用lab模型能更好的提取黄蓝色
def lab_Select(image,thresh = (212,220)):
    image[:,240:,:] = (0,0,0)
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
    lab_b = lab[:,:,2]
    if np.max(lab_b)>100:
            lab_b = lab_b/np.max(lab_b)*255
    binary = np.zeros_like(lab_b)
    binary[((lab_b>thresh[0])&(lab_b<thresh[1]))] = 1
    return binary
def line_color(image):
    hls_image = hlsSelect(image)
    lab_image = lab_Select(image)
    combined_binary = np.zeros_like(hls_image)
    combined_binary[(hls_image == 1)|(lab_image == 1)] = 1
    cv2.imshow('hls',hls_image)
    cv2.imshow('lab',lab_image)
    cv2.imshow('he',combined_binary)
    image =  dilate_erode(combined_binary,7)
    cv2.imshow('fp',image)
if __name__ == '__main__':
    # q_mqtt_data = queue.Queue(5)
    # mqtt_client = hqyj_mqtt.MQTTClient('127.0.0.1',21883,'bb','aa',q_mqtt_data)
    # i = 1
    # while  True:
    #     image = q_mqtt_data.get()
    #     if 'image' in image:
    #         a = base_64_up(image)
    #     cv2.imshow('image', a)
    #     key = cv2.waitKey(1)
    #     if key ==ord('q'):
    #         break
    #     elif key == ord('s'):
    #         cv2.imwrite(f'{i}.png',image)
    #         i += 1
    #         print('save successful')
    image =  cv2.imread('./2.png')
    image = change_eye(image)
    # image = get_line_tidu(image)
    # image = dilate_erode(image,5)
    line_color(image)
    # cv2.imshow('image',image)
    cv2.waitKey(0)


