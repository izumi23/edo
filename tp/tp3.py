import numpy as np
import pylab as plt
import scipy
from numpy import array
from scipy.integrate import odeint

plt.ion()
plt.show()


### Exo 2

plt.figure(0, figsize=(12, 10))
plt.clf()

Y0 = array([[2.*np.random.random() for j in range(2)] for i in range(10)])
eps = 0

def exo(eps, num):

    plt.subplot(2, 2, 2*num-1)
    plt.axis('scaled')
    plt.axis([-2, 2, -2, 2])

    f = lambda y, t: array([-y[1] - eps*y[0]*(y[0]*y[0] + y[1]*y[1]), y[0] - eps*y[1]*(y[0]*y[0] + y[1]*y[1])])

    t = np.linspace(0, 50, 10001)

    y = None

    for i in range(10):
        y0 = Y0[i,:]
        y = odeint(f, y0, t)
        plt.plot(y[:,0], y[:,1])

    lsp = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(lsp, lsp)
    DX, DY = f([X, Y], 0)
    N = np.hypot(DX, DY)
    plt.quiver(X, Y, DX/N, DY/N, N, angles='xy', scale=30)

    plt.subplot(2, 2, 2*num)
    plt.plot(t, y[:,0])

exo(0, 1)
exo(0.01, 2)


### Exo 3

plt.figure(1, figsize=(12, 10))
plt.clf()
plt.axis('scaled')
plt.axis([-4, 6, -3, 3])
plt.axhline(color='black')
plt.axvline(color='black')

T0 = array([-3.8, -2.8, -2.4, -2.2, -2.05, -2.0276440, -2.0276437, -1.9, -1.5, -1.0])
TF = array([-2.6, -1.2, -0.3, 0.2, 1.5, 4.5, 6.0, 6.0, 6.0, 6.0])

f = lambda y, t: y*y - t

for i in range(len(T0)):
    t = np.linspace(T0[i], TF[i], 10001)
    y = odeint(f, -3., t)
    plt.plot(t, y, color='blue')

T, Y = np.meshgrid(np.linspace(-4, 6, 31), np.linspace(-3, 3, 21))
DY = f(Y, T)
DT = np.ones_like(DY)
N = np.hypot(DT, DY)
plt.quiver(T, Y, DT, DY, N, angles='xy', scale=200)

T, Y = np.meshgrid(np.linspace(-4, 6, 1001), np.linspace(-3, 3, 1001))
DY = f(Y, T)
DT = np.ones_like(DY)
for A in [Y*Y-T, Y*Y-T+np.ones_like(T), Y*Y-T-np.ones_like(T)]:
    plt.contour(T, Y, A, 0, linestyles='solid', colors='red')


### Exo 4



































