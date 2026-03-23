import numpy as np
import matplotlib.pyplot as  plt
def  sigmoid(x):
    return  1/(1+np.exp(-x))
def  sigmoid_daoshu(x):
    return sigmoid(x)*(1-sigmoid(x))
x = np.linspace(-10,10,100)
y_sigmoid = sigmoid(x)
y_daoshu = sigmoid_daoshu(x)
plt.figure(figsize=(8,6))
plt.plot(x,y_sigmoid,label = 'sigmoid')
plt.plot(x,y_daoshu,label = 'daoshu')
plt.title("sigmoid")
plt.legend()
plt.grid(True)
plt.show()