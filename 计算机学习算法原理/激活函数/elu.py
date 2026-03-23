
import numpy as np
import matplotlib.pyplot as  plt
def  elu(x,a = 0.25):
    return  np.where(x>0,x,a*(np.exp(x)-1))
def  elu_daoshu(x,a=0.25):
    return np.where(x>0,1,elu(x)+a)
x = np.linspace(-10,10,100)
y_elu = elu(x)
y_derivative = elu_daoshu(x)
plt.figure(figsize=(8,6))
plt.plot(x,y_elu,label = 'elu',color = 'r')
plt.plot(x,y_derivative,label = 'daoshu',color = 'g')
plt.title("elu")
plt.legend()
plt.grid(True)
plt.show()