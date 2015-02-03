import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0.0, 5.0)
x2 = np.linspace(0.0, 2.0)

y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
y2 = np.cos(2 * np.pi * x2)

plt.title('A tale of 2 subplots')



x1 = np.linspace(0.0, 20.0, 100)
y1 = (np.e ** (-x1 / 4.)) * np.sin(x1)
plt.subplot(2, 2, 1)
plt.plot(x1, y1, 'black')
plt.title('$f(x) = e^{-x/4}\sin(x)$')
plt.ylabel('$f(x)$')
plt.grid()
plt.xlabel('$x$')

y2 = np.e**(-x1/4.)
plt.subplot(2, 2, 2)
plt.title('$f(x) = e^{-x/4}\sin(x)$')
plt.plot(x1, y1, 'red', x1, y2, '--')
plt.xlabel('time (s)')
plt.ylabel('$f(x)$')
plt.text(5, 0.4, '$e^{-x/4}$')
plt.xlabel('$x$')

y3 = np.e**(-x1) * (np.sin(x1))**2
plt.subplot(2, 2, 3)
plt.semilogy(x1, y3, 'g+')
plt.title('$f(x) = e^{-x}\sin(x)^2$')
plt.ylabel('$f(x)$')
plt.xlabel('$x$')


theta = np.linspace(0.0, 2 * np.pi, 1000)
flower = np.sin(-np.pi / 2 - theta * 6)
plt.subplot(2, 2, 4, polar=True)
plt.plot(theta, flower, 'violet')
plt.tight_layout()




plt.suptitle("'Close Enough' by Aaron Deich", size=18)
plt.show()
