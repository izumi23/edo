import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import odeint

plt.figure(0)
plt.clf()

a = 0.5

f = lambda y, t: np.array([y[0] - y[0]*y[0] - y[0]*y[1]/(1 + y[0]),
                           a*y[1] - y[1]*y[1]/(1 + y[0]*y[0])])

t = np.linspace(0, 20, 100001)

for i in range(20):
    y0 = np.array([1.5*np.random.random() for j in range(2)])
    y = odeint(f, y0, t)
    plt.plot(y[:,0], y[:,1], color='orange')

X, Y = np.meshgrid(np.linspace(0, 1.5, 16), np.linspace(0, 2.5, 26))
DX, DY = f([X, Y], 0)
N = np.hypot(DX, DY)
plt.quiver(X, Y, DX/N, DY/N, N, angles='xy', scale=20)

X, Y = np.meshgrid(np.linspace(0, 1.5, 1501), np.linspace(0, 2, 2501))
DX, DY = f([X, Y], 0)
plt.contour(X, Y, DX, 0, colors='blue')
plt.contour(X, Y, DY, 0, colors='green')

xs, ys = np.sqrt((1-a)/(1+a)), 2*a/(1+a)
plt.plot(xs, ys, marker='o', markersize=8, color='red')

t0 = 0
h = 0.00001
df = np.transpose(np.array([
    (f([xs+h, ys], t0) - f([xs-h, ys], t0)) / (2*h),
    (f([xs, ys+h], t0) - f([xs, ys-h], t0)) / (2*h),
]))

D, V = scipy.linalg.eig(df)

for i in range(2):
    if np.imag(D[i]) == 0:
        d = np.real(D[i])
        plt.plot([-1, 3], [d*(-1-xs)+ys, d*(3-xs)+ys], color='brown')


plt.axis('scaled')
plt.axis([-0.01, 1.5, -0.01, 1.5])
plt.title('a = {},    l1 = {:9.1f},    l2 = {:9.1f}'.format(a, D[0], D[1]))
plt.show()