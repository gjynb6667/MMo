
import numpy as np
import matplotlib.pyplot as  plt
def  lekyRelu(x,a = 0.25):
    return  np.where(x>0,x,a*x)
def  Leky_daoshu(x,a=0.25):
    return np.where(x>0,1,a)
x = np.linspace(-10,10,100)
y_lekyRelu = lekyRelu(x)
y_derivative = Leky_daoshu(x)
plt.figure(figsize=(8,6))
plt.plot(x,y_lekyRelu,label = 'LekyRelu',color = 'r')
plt.plot(x,y_derivative,label = 'daoshu',color = 'g')
plt.title("LekyRelu")
plt.legend()
plt.grid(True)
plt.show()