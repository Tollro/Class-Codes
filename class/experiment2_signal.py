import matplotlib.pyplot as plt
import numpy as np

#绘制连续余弦信号
x1 = np.linspace(-4*np.pi, 4*np.pi,1000)
sgl_cos = np.cos(x1)
plt.plot(x1,sgl_cos, label="cos(x)")
plt.legend()
plt.show()

#绘制离散周期信号
x2 = np.linspace(-5,5,11)
y0 = [5,3,1]
sgl_T = []

for i in range(11):
    sgl_T.append(y0[i%3])

plt.stem(x2,sgl_T,label="f(x)--T")
plt.grid(True)
plt.legend()
plt.show()