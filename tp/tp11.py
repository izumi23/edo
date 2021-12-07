import numpy as np
import scipy.interpolate, scipy.linalg
import matplotlib.pyplot as plt
from numpy import array, cos, sin, pi, exp, log10, abs
from scipy.integrate import quad, odeint
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
vx = aux[:, 0]
vy = aux[:, 1]

L = np.tensordot(vx, np.ones_like(vx), 0)
L1 = np.tensordot(np.ones_like(vx), vx, 0)
K = abs(L - L1)

R = np.zeros((n+2, n+2), dtype=float)
R[:n, :n] = K
R[n, :n] = 1
R[n+1, :n] = vx
R[:n, n] = 1
R[:n, n+1] = np.transpose(vx)

B = np.zeros((n+2), dtype=float)
B[:n] = vy

X = solve(R, B)
alpha = X[:-2]
beta = X[-2:]

x = np.linspace(np.min(vx), np.max(vx), int((np.max(vx)-np.min(vx))/0.005)+1)

plt.figure(1)
plt.plot(vx, vy, linestyle='None', marker='o')
plt.plot(x, beta[0]+beta[1]*x)
plt.plot(vx, beta[0]+beta[1]*vx + np.sum(K, 1))









