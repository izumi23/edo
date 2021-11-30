import numpy as np
import scipy.interpolate, scipy.linalg
import matplotlib.pyplot as plt
from numpy import array, cos, pi

plt.ion()
plt.show()


## Exercice 1

lx = array([-1, 0, 1, 1.5, 3])
ly = array([1, 0, 2, 1, 0])
vx = np.linspace(-1, 3, 501)
Pl = 1/72 * array([50, -195, 58, 231, 0])

plt.figure(0)
plt.clf()
plt.title("Plusieurs méthodes d'interpolation")
plt.xlabel("x")
plt.plot(lx, ly, linestyle='None', marker='o', label="Points d'interpolation")
plt.plot(vx, np.polyval(Pl, vx), label="lagrange")

for kind in ['linear', 'nearest', 'cubic']:
    P = scipy.interpolate.interp1d(lx, ly, kind=kind)
    plt.plot(vx, P(vx), label=kind)

V = np.vander(lx)
Pv = scipy.linalg.solve(V, ly)
plt.plot(vx, np.polyval(Pv, vx), label="vander")

Pf = np.polyfit(lx, ly, 4)
plt.plot(vx, np.polyval(Pf, vx), label="polyfit")

plt.legend()

plt.figure(1)
plt.clf()
plt.title("Différences entre méthodes d'interpolation")
plt.xlabel("x")
plt.plot(vx, np.polyval(Pv - Pl, vx), label="vander - lagrange")
plt.plot(vx, np.polyval(Pf - Pl, vx), label="polyfit - lagrange")


## Exercice 2

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


## Exercice 3

for i, a in enumerate([0.1, 1], 4):

    plt.figure(i)
    plt.clf()
    f = lambda x: 1/(x*x + a)
    vx = np.linspace(-1, 1, 501)

    for i in range(1, 9):
        n = i
        if i > 6:
            n += 12
        lx = array([cos((k + 0.5)*pi/(n + 1)) for k in range(n+1)])
        ly = f(lx)
        P = np.polyfit(lx, ly, n)
        plt.subplot(2, 4, i)
        plt.title("n = {}".format(n))
        plt.plot(vx, f(vx))
        plt.plot(vx, np.polyval(P, vx))
        plt.plot(lx, ly, linestyle='None', marker='o')


## Exercice 4

def bracket(lx, ly):
    n = len(lx)
    A = ly.copy()
    for k in range(1, n):
        for i in range(n-1, k-1, -1):
            A[i] = (A[i] - A[i-1])/(lx[i] - lx[i-k])
    return A

lx = array([-1, 0, 1, 1.5, 3])
ly = array([1., 0, 2, 1, 0])
vx = np.linspace(-1, 3, 501)

def interp_dd_val(lx, ly, x):
    A = bracket(lx, ly)
    y = A[0]
    v = 1
    for i in range(len(lx)-1):
        v *= x - lx[i]
        y += A[i+1]*v
    return y

def interp_dd(lx, ly, vx):
    return array([interp_dd_val(lx, ly, x) for x in vx])

plt.figure(1)
plt.plot(vx, interp_dd(lx, ly, vx) - np.polyval(Pl, vx), label="difdiv - lagrange")
plt.legend()






















