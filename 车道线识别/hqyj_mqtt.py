import json
import paho.mqtt.client as mqtt
import time
import threading


class MQTTClient:
    """
    mqtt
    """

    def __init__(self, target_ip, default_port, sub_topic, pub_topic, q_mqtt_data):
        """
        初始化
        :param target_ip: mqtt服务器IP
        :param default_port: mqtt端口号
        :param sub_topic: sub的topic
        :param pub_topic: pub的topic
        :param q_mqtt_data: 消息队列
        """
        # 创建一个MQTT客户端
        self.client = mqtt.Client()
        # 使用消息队列传输数据
        self.q_mqtt_data = q_mqtt_data
        # 绑定连接函数和接收消息函数
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        # 存储broker的IP地址
        self.target_ip = target_ip
        # 端口号
        self.default_port = default_port
        # sub主题
        self.sub_topic = sub_topic
        # pub主题
        self.pub_topic = pub_topic
        # 初始化连接状态为False，表示未连接
        self.connected = False
        # 尝试连接MQTT代理
        self.connect()
        # 初始化一个空字符串，用于存储接收到的消息内容
        self.message = ""

    def connect(self):
        """
        连接MQTT服务器
        """
        try:
            self.client.connect(self.target_ip, self.default_port, 5)
            self.connected = True
            self.client.subscribe(self.sub_topic, qos=0)
            self.client.loop_start()
        except:
            self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        """
        连接成功回调函数
        :param client:
        :param userdata:
        :param flags:
        :param rc:
        :return:
        """
        if rc == 0:
            self.connected = True
            print("Connected to MQTT OK!")
        else:
            self.connected = False
            print("Failed to connect, return code %d\n", rc)

    def on_message(self, client, userdata, msg):
        """
        接收数据回调函数
        :param client:
        :param userdata:
        :param msg: 接收到的信息
        :return: none
        """
        try:
            self.q_mqtt_data.put(json.loads(msg.payload))
        except Exception as e:
            print(e)

    def send_mqtt(self, data):
        """
        发送mqtt消息
        :param data: 发送的数据
        :return: none
        """
        if self.connected:
            self.client.publish(self.pub_topic, payload=data, qos=0)
        else:
            print("Not connected to MQTT server. Message not sent.")

    def on_disconnect(self, client, userdata, rc):
        """
        连接断开回调函数
        """
        print("Disconnected from MQTT server. Reconnecting...")
        self.connected = False
        threading.Thread(target=self.reconnect)

    def reconnect(self):
        """
        重新连接MQTT服务器
        """
        while not self.connected:
            self.connect()
            time.sleep(5)
