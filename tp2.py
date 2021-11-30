import numpy as np
import pylab as plt
import scipy
from numpy import array
from scipy.integrate import odeint

plt.ion()
plt.show()

plt.figure(0)
plt.clf()
plt.axis([0., 5., -0.01, 3.])

f = lambda y, t: y*(y - 1.)*(2. - y)

t = np.linspace(0., 5., 501)

for y0 in [0., 0.2, 0.7, 0.95, 1.0, 1.1, 1.6, 2., 2.3, 3.]:
    y = odeint(f, y0, t)
    plt.plot(t, y)

T, Y = np.meshgrid(np.arange(0, 5, 0.2), np.arange(0, 3, 0.2))
DY = f(Y, 0)
DT = np.ones_like(T)
N = np.hypot(DT, DY)
plt.quiver(T, Y, DT/N, DY/N, N, angles='xy', scale=30)


def exo(a, b, c, d, fig):

    plt.figure(fig)
    plt.clf()
    plt.axis('scaled')
    plt.axis([-4, 4, -4, 4])

    f = lambda y, t: array([a*y[0] + b*y[1], c*y[0] + d*y[1]])

    t = np.linspace(0., 10, 1001)
    t1 = np.flipud(t.copy())

    for i in range(15):
        y0 = array([4.*np.random.random() - 2. for k in range(2)])

        y = odeint(f, y0, t)
        plt.plot(y[:,0], y[:,1], color='blue')

        y = odeint(f, y0, t1)
        plt.plot(y[:,0], y[:,1], color='red')

    X, Y = np.meshgrid(np.linspace(-4, 4, 25), np.linspace(-4, 4, 25))
    DX, DY = f([X, Y], 0)
    N = np.hypot(DX, DY)
    plt.quiver(X, Y, DX, DY, N, angles='xy', scale=1000)

    X, Y = np.meshgrid(np.linspace(-4, 4, 1001), np.linspace(-4, 4, 1001))
    DX, DY = f([X, Y], 0)
    plt.contour(X, Y, DX, 0, colors='lime')
    plt.contour(X, Y, DY, 0, colors='green')

    A = array([[a, b], [c, d]])
    D, V = scipy.linalg.eig(A)
    r = array([-4, 4])
    plt.plot(r, V[1, 0]/V[0, 0]*r, color='orange')
    plt.plot(r, V[1, 1]/V[0, 1]*r, color='pink')

    plt.title('l1 = {:9.1f},    l2 = {:9.1f}'.format(D[0], D[1]))


exo(10, -6, 3, -9, 1)
exo(-4, 2, 1, -5, 2)
exo(1, -5, 1, -1, 3)
exo(0.5, 5, -2, -3, 4)







