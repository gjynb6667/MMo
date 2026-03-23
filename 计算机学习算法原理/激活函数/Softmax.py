import numpy as np
import matplotlib.pyplot as  plt
def  Softmax(x):
    vals = np.exp(x)
    return  vals/np.sum(vals)
def  Softmax_daoshu(x):
    s = Softmax(x)
    return np.diagflat(s) - np.outer(s,s)
x = np.linspace(-5,5,100)
y_softmax = Softmax(x)
y_derivative = Softmax_daoshu(x)
plt.figure(figsize=(8,6))
# plt.plot(x,y_softmax,label = 'Softmax')
plt.plot(x,y_derivative,label = 'daoshu')
plt.title("Softmax")
# plt.legend()
# plt.grid(True)
plt.show()