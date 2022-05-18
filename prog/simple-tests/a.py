import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5,5,0.1)
x = np.arange(-10,10,0.1)
noise = np.random.randn(len(x))

a = np.arange(-2,2,0.1)
b = 0

def fn(x,a,b):
    return a * x + b

loss_list = []
for v in a:

    y = fn(x,v,b)
    y_hat = y*noise

    loss = ((y_hat - y)**2).mean()
    loss_list.append(loss)

plt.plot(a, loss_list)
plt.show()

# compute loss based on different data but fixed parameters
x = np.arange(1,2,0.1)
noise = np.random.randn(len(x))
print(len(x))

a = 2
y = fn(x, a, b)
y_hat = y*noise
loss = ((y_hat - y)**2).mean()
print(loss)

x = x[0:3]
noise = noise[0:3]
print(len(x))
y = fn(x, a, b)
y_hat = y*noise
loss = ((y_hat - y)**2).mean()
print(loss)
