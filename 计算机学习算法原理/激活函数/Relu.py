import numpy as np
import matplotlib.pyplot as  plt
def  Relu(x):
    return  np.maximum(0,x)
def  Relu_daoshu(x):
    return np.where(x>0,1,0)
x = np.linspace(-10,10,100)
y_reLU = Relu(x)
y_derivative = Relu_daoshu(x)
plt.figure(figsize=(8,6))
plt.plot(x,y_reLU,label = 'Relu')
plt.plot(x,y_derivative,label = 'daoshu')
plt.title("Relu")
plt.legend()
plt.grid(True)
plt.show()