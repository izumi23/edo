import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.integrate import odeint
from numpy import array, exp, log, sqrt

plt.ion()
plt.show()


## Exercice 1

def rungekutta(f, y0, vt):
    y = y0
    h = vt[1] - vt[0]
    vy = [y]
    for t in vt[:-1]:
        p1 = f(t, y)
        p2 = f(t + h/2, y + h/2*p1)
        p3 = f(t + h/2, y + h/2*p2)
        p4 = f(t + h, y + h*p3)
        y = y + h/6*(p1 + 2*p2 + 2*p3 + p4)
        vy.append(y)
    return array(vy)

r, K = 1, 1
y0 = 0.1
t0, tf = 0, 2
h = 0.25

f = lambda t, y: r*y*(1 - y/K)

y_sol = lambda t, y0: y0*K / (y0 - exp(-r*t)*(-K + y0))

vt = np.arange(t0, tf+h, h)

plt.figure(0)
plt.clf()
plt.title("Modèle logistique")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.plot(vt, y_sol(vt, y0), color='g', label="Solution exacte")

H = []
E = []

for k in range(1, 11):
    h = 2**-k
    H.append(h)
    vt = np.arange(t0, tf+h, h)
    vy = rungekutta(f, y0, vt)

    if k == 2:
        plt.plot(vt, vy, marker='.', linestyle='None', color='orange',
                 label="RK4, h = {}".format(h))
        plt.legend()

    e = norm(vy - y_sol(vt, y0), ord=np.inf)
    E.append(e)

plt.figure(1)
plt.clf()
plt.xlabel("Pas h")
plt.ylabel("Erreur globale e")
[a, b] = np.polyfit(log(H), log(E), 1)
plt.loglog(H, exp(a*log(H) + b), color='g', label="y = ax")
plt.loglog(H, E, linestyle="None", marker='.', color='orange', label='Données expérimentales')
plt.legend()
plt.title("log e ~ a log h,  a = {:.2f}".format(a))


## Exercice 2

f = lambda t, y: array([y[0]*(1 - y[1]), y[1]*(2*y[0] - 1.5)])

y0 = [2, 2]
t0, tf = 0, 15
plt.figure(2, figsize=(12, 4))
plt.clf()
plt.suptitle("Modèle de Lotka-Volterra")

for i, h in enumerate([0.05, 0.005, 0.2]):

    t = np.arange(t0, tf+h, h)

    f_sym = lambda y, t: f(t, y)
    y_sol = odeint(f_sym, y0, np.arange(t0, tf+0.005, 0.005))
    y = rungekutta(f, y0, t)

    plt.subplot(1, 3, i+1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(y_sol[:,0], y_sol[:,1], color='g', label="odeint")
    plt.plot(y[:,0], y[:,1], marker='.', markersize=2, linestyle='None',
             color='orange', label="RK4, h = {}".format(h))
    plt.legend()










