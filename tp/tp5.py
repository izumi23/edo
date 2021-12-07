import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.integrate import odeint
from numpy import array, exp, log

plt.ion()
plt.show()


## Exercice 1

def euler(f, y0, vt):
    y = y0
    h = vt[1] - vt[0]
    vy = [y]
    for t in vt[:-1]:
        y = y + h*f(t, y)
        vy.append(y)
    return array(vy)

r, K = 1, 1
y0 = 0.1
t0, tf = 0, 2

f = lambda t, y: r*y*(1 - y/K)
y_sol = lambda t: y0*K/(y0 - exp(-r*t)*(-K + y0))
vt = np.arange(t0, tf+0.25, 0.25)

plt.figure(0)
plt.clf()
plt.plot(vt, y_sol(vt), color='g', label="Solution exacte")

H = []
E = []

for k in range(1, 11):
    h = 2**-k
    H.append(h)
    vt = np.arange(t0, tf+h, h)
    vy = euler(f, y0, vt)

    if k == 2:
        plt.plot(vt, vy, marker='.', color='orange', label="h = {}".format(h))
        plt.legend()

    e = norm(vy - y_sol(vt), ord=np.inf)
    E.append(e)

plt.figure(1)
plt.clf()
plt.xlabel("Pas h")
plt.ylabel("Erreur globale e")
[a, b] = np.polyfit(log(H), log(E), 1)
plt.loglog(H, exp(a*log(H) + b), color='g', label="a = {:.6f}".format(a))
plt.loglog(H, E, linestyle="None", marker='.', color='orange')
plt.legend()
plt.title("log e ~ a log h")


## Exercice 2

f = lambda t, y: array([y[0]*(1 - y[1]), y[1]*(2*y[0] - 1.5)])
f_sym = lambda y, t: f(t, y)

y0 = array([2, 2])

t = np.arange(0, 15.005, 0.005)
y = odeint(f_sym, y0, t)
plt.figure(2)
plt.clf()
plt.xlabel("Proies")
plt.ylabel("Prédateurs")
plt.plot(y[:,0], y[:,1], color='g', label="Solution exacte")

for h in [0.05, 0.005]:
    vt = np.arange(0, 15.001, h)
    vy = euler(f, y0, vt)
    plt.plot(vy[:,0], vy[:,1], marker='.', markersize=2, linestyle='None',
             label="Euler, h = {}".format(h))

plt.legend()





























