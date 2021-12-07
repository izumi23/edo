import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.integrate import odeint
from numpy import array, exp, log, sqrt

plt.ion()
plt.show()


## Exercice 1

def heun(f, y0, vt):
    y = y0
    h = vt[1] - vt[0]
    vy = [y]
    for t in vt[:-1]:
        p1 = f(t, y)
        p2 = f(t + h, y + h*p1)
        y = y + h/2*(p1 + p2)
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
    vy = heun(f, y0, vt)

    if k == 2:
        plt.plot(vt, vy, marker='.', linestyle='None', color='orange',
                 label="Heun, h = {}".format(h))
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
plt.title("log e ~ a log h,  a = {:.6f}".format(a))


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
    y = heun(f, y0, t)

    plt.subplot(1, 3, i+1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(y_sol[:,0], y_sol[:,1], color='g', label="odeint")
    plt.plot(y[:,0], y[:,1], marker='.', markersize=2, linestyle='None',
             color='orange', label="Heun, h = {}".format(h))
    plt.legend()



## Exercice 4

def adapt12(f, y0, t0, tf, Tol):
    hmin = 1e-7
    hmax = (tf - t0)/2
    h, y, t = hmax, y0, t0
    vt = [t]
    vy = [y]
    while tf-t > hmin/2:
        Ea = h/2*norm(f(t + h, y + h*f(t, y)) - f(t, y))
        u = sqrt(Ea/Tol)
        if u < 1:
            y = y + h*f(t, y)
            t = t + h
            vt.append(t)
            vy.append(y)
            h = min(hmax, h/u)
        else:
            h = max(hmin, 0.95*h/u)
    return array(vt), array(vy)

def adapt21(f, y0, t0, tf, Tol):
    hmin = 1e-7
    hmax = (tf - t0)/2
    h, y, t = hmax, y0, t0
    vt = [t]
    vy = [y]
    while tf-t > hmin/2:
        Ea = h/2*norm(f(t + h, y + h*f(t, y)) - f(t, y))
        u = sqrt(Ea/Tol)
        if u < 1:
            y = y + h/2*(f(t, y) + f(t + h, y + h*f(t, y)))
            t = t + h
            vt.append(t)
            vy.append(y)
            h = min(hmax, h/u)
        else:
            h = max(hmin, 0.95*h/u)
    return array(vt), array(vy)


Tol = 0.01

fig_nb = 3

for a in [5, 14]:

    f = lambda t, y: array([y[1], -y[0] + a*(1 - y[0]**2)*y[1]])
    f_sym = lambda y, t: f(t, y)

    y0 = array([1, 1])
    t0, tf = 0, 5*a
    t = np.arange(t0, tf+0.01, 0.01)
    y, out = odeint(f_sym, y0, t, full_output=True)

    for adapt in [adapt12, adapt21, heun]:
        vt, vy = None, None
        if adapt == heun:
            vt, vy = t, heun(f, y0, t)
        else:
            vt, vy = adapt(f, y0, t0, tf, Tol)

        plt.figure(fig_nb, figsize=(12, 10))
        fig_nb += 1
        plt.clf()
        plt.suptitle("Système de Van der Pol,  a = {}".format(a))

        legends = ["x(t)", "y(t)"]
        for j in range(2):
            plt.subplot(3, 1, 1+j)
            plt.xlabel("t")
            plt.ylabel(legends[j])
            plt.plot(t, y[:,j], color='g', label="odeint")
            plt.plot(vt, vy[:,j], color='orange', linestyle='None',
                     marker='.', markersize=2, label=adapt.__name__)
            plt.legend()

        plt.subplot(3, 1, 3)
        plt.xlabel("t")
        plt.ylabel("Pas de temps")
        plt.semilogy(t[:-1], out['hu'], label="odeint", color='g')
        plt.semilogy(vt[:-1], vt[1:] - vt[:-1], color='orange',
            linestyle='None', marker='.', markersize=2, label=adapt.__name__)
        plt.legend()

























