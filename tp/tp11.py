import numpy as np
import scipy.interpolate, scipy.linalg
import matplotlib.pyplot as plt
from numpy import array, cos, sin, pi, exp, log10, abs, min, max, sum
from scipy.integrate import quad, odeint
from scipy.interpolate import interp1d
from scipy.linalg import norm, solve
from os import chdir

plt.ion()
plt.show()

## Exercice 1

a, b = 0, 1
M = 49
h = (b-a)/(M+1)
sigma, nu = 10, 0.1
ua, ub = 2.0, 0.2
w = lambda x: 2 - 1.8*x
vx = np.linspace(a, b, M)
y0 = w(vx)
k = 0.05
T = 1
vt = np.arange(0, T+k, k)

A = nu/h**2 * (-2*np.eye(M) + np.eye(M, k=1) + np.eye(M, k=-1))
B = np.zeros((M))
B[0], B[-1] = nu/h**2*ua, nu/h**2*ub

f = lambda y, t: A @ y + B - sigma*y/(1 + y)

# vysol = odeint(f, y0, vt)
# plt.figure(4)
# plt.clf()
# for i, t in enumerate(vt):
#     plt.clf()
#     plt.plot(vx, vy[i, :])
#     plt.axis([0, 1, -0.05, 2.05])
#     plt.title("t = {:.2f}".format(t))
#     plt.pause(1)

# plt.figure(4)
# plt.clf()
# y = y0
# for t in vt[:-1]:
#     P = np.diag([1/(1 + y[j]) for j in range(M)])
#     y = solve(np.eye(M) - k*(A - sigma*P), y + k*B)
#     plt.clf()
#     plt.plot(vx, y, linestyle='None', marker='o', markersize=3)
#     plt.axis([0, 1, -0.05, 2.05])
#     plt.title("t = {:.2f}".format(t))
#     plt.pause(1)


## Exercice 2

F = lambda U: -A @ U + sigma * U/(1 + U)
dF = lambda U: -A + sigma * U/(1 + U)**2

U = y0
for n in range(1, 11):
    err = solve(dF(U), F(U))
    #print(n, norm(err))
    U = U - err
#print(U)


## Exercice 3

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
    plt.plot(vx, beta[0]+beta[1]*vx, label="d??rive")
    plt.plot(vx, u(vx), label="krigeage")

    if fig == 1:
        plt.plot(vx, (interp1d(lx, ly, kind='cubic'))(vx), label="interp1d")
    if dirac:
        plt.plot(lx, np.polyval(np.polyfit(lx, ly, 1), lx), label="R??gression affine")
    plt.legend()


## Exercice 4

c1, c2 = 1., 3.
M = 49
h = 1/(M + 1)
T = 60
k = 20*h
alpha, beta = 4e-4, 2e-3
u, v = c1, c2/c1

I = np.eye(M)
D = 1/h**2 * (-2*np.diag(M) + np.diag(M, 1) + np.diag(M, -1)
A = np.zeros((2*M, 2*M))





























