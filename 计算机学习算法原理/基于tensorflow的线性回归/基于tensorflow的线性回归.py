import numpy as np
import torch
import  tensorflow as tf
from  tensorflow.keras import Model
data = [[-0.5, 7.7], [1.8, 98.5], [0.9, 57.8], [0.4, 39.2], [-1.4, -15.7], [-1.4, -37.3], [-1.8, -49.1], [1.5, 75.6], [0.4, 34.0], [0.8, 62.3]]
data = np.array(data)
x_data = data[:,0]
y_data = data[:,1]
# x_tensor = tf.constant(x_data,dtype=tf.float32)
x_tensor = tf.constant(np.expand_dims(x_data,axis=1),dtype = tf.float32)
y_tensor = tf.constant(y_data, dtype= tf.float32)
dataset = tf.data.Dataset.from_tensor_slices((x_tensor,y_tensor))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(5)
dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# for item in dataset:
#     print(item)
# model =  tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape = (1,))])
# class Linear(Model):
#     def __init__(self):
#         super(Linear,self).__init__()
#         self.linear = tf.keras.layers.Dense(1)
#     def call(self, x, **kwargs):
#         x = self.linear(x)
#         return  x
def linear():
    input = tf.keras.layers.Input(shape=(1,),dtype=tf.float32)
    y = tf.keras.layers.Dense(1)(input)
    model = tf.keras.models.Model(inputs = input , outputs = y)
    return  model
model = linear()
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01)
model.compile(optimizer = optimizer, loss = "mean_squared_error")
epoches = 500
for epoche in range(1,epoches+1):
    total_loss = 0
    for batch_x,batch_y in dataset:
        history = model.fit(x_tensor, y_tensor)
        loss = history.history["loss"][0]
        total_loss +=loss
        print(loss)
    if epoche % 10 == 0 or epoche == 1:
        print(f'epoch:{epoche},loss:{loss}')
