import numpy as np
import matplotlib.pyplot as  plt
def  tanh(x):
    return  np.tanh(x)
def  tanh_daoshu(x):
    return 1- np.tanh(x)**2
x = np.linspace(-10,10,100)
y_tanh = tanh(x)
y_derivative = tanh_daoshu(x)
plt.figure(figsize=(8,6))
plt.plot(x,y_tanh,label = 'tanh')
plt.plot(x,y_derivative,label = 'daoshu')
plt.title("tanh")
plt.legend()
plt.grid(True)
plt.show()