import numpy as np, matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(-5, 5, 0.1)
y = softmax(x)      

plt.plot(x,y)
plt.grid()
plt.show()