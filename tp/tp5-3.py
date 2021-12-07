import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.integrate import odeint
from numpy import array, exp, log

plt.ion()
plt.show()

## Question 1

a = 3
f = lambda y, t: array([y[1], -y[0] + a*(1 - y[0]**2)*y[1]])

plt.figure(0, figsize=(7, 10))
plt.clf()
plt.axis('scaled')
plt.axis([-2.5, 2.5, -6, 6])

t = np.linspace(0, 5*a, 1001)
CI = -2 + 4*np.random.rand(10, 2)

for y0 in CI:
    y = odeint(f, y0, t)
    plt.plot(y[:,0], y[:,1], color='orange')

X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 13), np.linspace(-6, 6, 31))
DX, DY = f([X, Y], 0)
N = np.hypot(DX, DY)
plt.quiver(X, Y, DX/N, DY/N, N, angles='xy', scale=20)

X, Y = np.meshgrid(np.linspace(-2.5, 2.5, 1001), np.linspace(-6, 6, 1001))
DX, DY = f([X, Y], 0)
plt.contour(X, Y, DX, 1, colors='b')
plt.contour(X, Y, DY, 1, colors='g')


## Question 2

for i, a in enumerate([3, 14]):

    f = lambda y, t: array([y[1], -y[0] + a*(1 - y[0]**2)*y[1]])
    y0 = (1, 0)
    t = np.arange(0, 5*a, 0.01)

    y, out = odeint(f, y0, t, full_output=True)

    plt.figure(i+1, figsize=(10, 10))
    plt.clf()

    plt.subplot(4, 1, 1)
    plt.plot(t, y[:,0], label="\u03B8(t)")
    plt.legend(loc= 'upper right')

    plt.subplot(4, 1, 2)
    plt.plot(t, y[:,1], label="\u03B8'(t)", color='g')
    plt.legend(loc= 'upper right')

    plt.subplot(4, 1, 3)
    plt.semilogy(t[:-1], out['hu'], label="pas(t)", color='orange')
    plt.legend(loc= 'upper right')

    plt.subplot(4, 1, 4)
    plt.plot(t[:-1], out['nqu'], label="ordre(t)", color='red')
    plt.legend(loc= 'upper right')

    plt.suptitle("a = {}".format(a))




























