import  hqyj_mqtt
import  queue
import  base64
import  numpy as  np
import cv2
def base_64_up(image):
    a = base64.b64decode(image['image'])
    b = np.frombuffer(a,dtype= np.uint8)
    c = cv2.imdecode(b,cv2.IMREAD_COLOR)
    return  c
if __name__ == '__main__':
    q_mqtt_data = queue.Queue(5)
    mqtt_client = hqyj_mqtt.MQTTClient('127.0.0.1',21883,'bb','aa',q_mqtt_data)

    while  True:
        image = q_mqtt_data.get()
        if 'image' in image:
            a = base_64_up(image)
        cv2.imshow('image', a)
        key = cv2.waitKey(1)
        if key ==ord('q'):
            break

