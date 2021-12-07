## Watanabe Izumi

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numpy import array

plt.ion()
plt.show()

for i, a in enumerate([0.1, 1], 2):

    plt.figure(i)
    plt.clf()
    f = lambda x: 1/(x*x + a)
    vx = np.linspace(-1, 1, 501)

    for i in range(1, 9):
        n = i
        if i > 6:
            n += 12
        lx = np.linspace(-1, 1, n+1)
        ly = f(lx)
        P = np.polyfit(lx, ly, n)
        plt.subplot(2, 4, i)
        plt.title("n = {}".format(n))
        plt.plot(vx, f(vx))
        plt.plot(vx, np.polyval(P, vx))
        plt.plot(lx, ly, linestyle='None', marker='o')
