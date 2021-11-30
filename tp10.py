import numpy as np
import scipy.interpolate, scipy.linalg
import matplotlib.pyplot as plt
from numpy import array, cos, sin, pi, exp, log10, abs
from scipy.integrate import quad, odeint

plt.ion()
plt.show()

## Exercice 1

# f = lambda x: exp(x)*(1/3 - x)*(cos(10*x))**2
# a, b = 1, 2.5

f = lambda x: exp(-1/x)*sin(3*x)/x**2
a, b = 0.5, 1

I_exact, _ = quad(f, a, b, epsabs=1e-14)

def trapezes(m):
    A = np.ones((m+1), dtype=float)
    A[0] = A[-1] = 0.5
    return A

def simpson(m):
    A = 2 * np.ones((m+1), dtype=float)
    for k in range(1, m+1, 2):
        A[k] = 4
    A[0] = A[-1] = 1
    return 1/3 * A

def pointmilieu(m):
    A = 2 * np.ones((m+1), dtype=float)
    for k in range(0, m+1, 2):
        A[k] = 0
    return A

vm = None
vJ = []

for u, method in enumerate([trapezes, simpson, pointmilieu]):

    vm = []
    verr = []

    for k in range(3, 13):
        m = 2**k
        h = (b-a)/m
        vx = np.arange(a, b+h, h)
        I = h * sum(f(vx) * method(m))
        if u == 2:
            vJ.append(I)
        vm.append(m)
        verr.append(abs(I - I_exact))

    vm = array(vm)
    verr = array(verr)

    [p1, p0] = np.polyfit(log10(vm), log10(verr), 1)

    plt.figure(u)
    plt.clf()
    plt.xlabel("m")
    plt.ylabel("|I - I_h|")
    plt.loglog(vm, 10**(p1*log10(vm) + p0), label="log |I - I_h| ~ a log m")
    plt.loglog(vm, verr, linestyle='None', marker='o', label="Données")
    plt.legend()
    plt.title("Méthode de {},  a = {:.2f}".format(method.__name__, p1))

vJ = array(vJ)
vS = 1/3 * (4 * vJ[1:] - vJ[:-1])
vm = vm[1:]
verr = abs(I_exact - vS)

[p1, p0] = np.polyfit(log10(vm), log10(verr), 1)

plt.figure(3)
plt.clf()
plt.xlabel("m")
plt.ylabel("|I - I_h|")
plt.loglog(vm, 10**(p1*log10(vm) + p0), label="log |I - I_h| ~ a log m")
plt.loglog(vm, verr, linestyle='None', marker='o', label="Données")
plt.legend()
plt.title("Méthode de romberg,  a = {:.2f}".format(p1))


## Exercice 3

a, b = 0, 1
M = 49
h = (b-a)/(M+1)
sigma, nu = 10, 0.1
ua, ub = 2.0, 0.2
w = lambda x: 2 - 1.8*x
k = 0.05
T = 1

A = -2*np.eye(M) + np.eye(M, k=1) + np.eye(M, k=-1)
B = np.zeros((M))
B[0], B[-1] = ua, ub

f = lambda y, t: nu/h**2 * (A @ y + B) - sigma*y/(1 + y)

vx = np.linspace(a, b, M)
vt = np.arange(0, T+k, k)
y = odeint(f, w(vx), vt)

plt.figure(4)
plt.clf()
for i, t in enumerate(vt):
    plt.clf()
    plt.plot(vx, y[i, :])
    plt.axis([0, 1, -0.05, 2.05])
    plt.title("t = {:.2f}".format(t))
    plt.pause(1)

























