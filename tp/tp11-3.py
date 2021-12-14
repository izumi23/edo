## WATANABE Izumi

import numpy as np
import matplotlib.pyplot as plt
from numpy import array, cos, sin, pi, exp, log10, abs, min, max, sum
from scipy.interpolate import interp1d
from scipy.linalg import norm, solve
from os import chdir

plt.ion()
plt.show()

chdir('/users/dptmaths/watanabe/edo/tp/')
aux = np.loadtxt('krigeage.txt')
n = len(aux)
lx = aux[:, 0]
ly = aux[:, 1]

vg = [(1, "affine", False), (3, "cubique", False), (3, "cubique+dirac", True)]

for fig, (p, inttype, dirac) in enumerate(vg):

    g = np.vectorize(lambda x: x**p + (dirac and x==0)*5000)
    L = np.tensordot(lx, np.ones_like(lx), 0)
    L1 = np.tensordot(np.ones_like(lx), lx, 0)
    K = g(abs(L - L1))

    R = np.zeros((n+2, n+2), dtype=float)
    R[:n, :n] = K
    R[n, :n] = 1
    R[n+1, :n] = lx
    R[:n, n] = 1
    R[:n, n+1] = np.transpose(lx)

    B = np.zeros((n+2), dtype=float)
    B[:n] = ly

    X = solve(R, B)
    alpha = X[:-2]
    beta = X[-2:]

    h = 0.005
    vx = np.arange(min(lx), max(lx)+h, h)
    u = lambda x: beta[0] + beta[1]*x + sum(alpha * g(abs(x - lx)))
    u = np.vectorize(u)

    plt.figure(fig)
    plt.clf()
    plt.title("Interpolation par krigeage, " + inttype)
    plt.plot(lx, ly, linestyle='None', marker='o', label="Points d'interpolation")
    plt.plot(vx, beta[0]+beta[1]*vx, label="dérive")
    plt.plot(vx, u(vx), label="krigeage")

    if fig == 1:
        plt.plot(vx, (interp1d(lx, ly, kind='cubic'))(vx), label="interp1d")
    if dirac:
        plt.plot(lx, np.polyval(np.polyfit(lx, ly, 1), lx), label="Régression affine")
    plt.legend()



