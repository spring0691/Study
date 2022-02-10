import numpy as np, matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5,5, 0.1)
y = relu(x)

plt.plot(x,y)
plt.grid()
plt.show()