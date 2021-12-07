# WATANABE Izumi

import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import cos, sin
from scipy.integrate import odeint

for i, a in enumerate([0, 0.2, -0.2]):

    plt.figure(i)
    plt.clf()
    plt.title("WATANABE Izumi.    a = {:.1f}".format(a))
    plt.axis('scaled')
    plt.axis([-2, 2, -2, 2])

    f = lambda y, t: np.array([cos(a)*(y[1] - y[0]**3 + y[0]) + sin(a)*y[0],
                            sin(a)*(y[1] - y[0]**3 + y[0]) - cos(a)*y[0]])

    t = np.linspace(0, 20, 1001)

    for i in range(10):
        y0 = np.array([-2 + 4*np.random.random() for j in range(2)])
        y = odeint(f, y0, t)
        plt.plot(y[:,0], y[:,1], color='orange')

    X, Y = np.meshgrid(np.linspace(-2, 2, 21), np.linspace(-2, 2, 21))
    DX, DY = f([X, Y], 0)
    N = np.hypot(DX, DY)
    plt.quiver(X, Y, DX/N, DY/N, N, angles='xy', scale=20)

    X, Y = np.meshgrid(np.linspace(-2, 2, 1001), np.linspace(-2, 2, 1001))
    DX, DY = f([X, Y], 0)
    plt.contour(X, Y, DX, 0, colors='blue')
    plt.contour(X, Y, DY, 0, colors='green')

    plt.show()
