import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm, solve
from scipy.integrate import odeint
from numpy import array, exp, log

plt.ion()
plt.show()


## Watanabe - TP 7, exercice 2

def euler_imp(f, df, y0, vt):

    d = len(y0)
    y = y0
    h = vt[1] - vt[0]
    vy = [y]
    for i in range(len(vt) - 1):

        t = vt[i]
        z = y
        E = []
        for n in range(1, 11):
            F = z - y - h*f(t+h, z)
            dF = np.eye(d) - h*df(t+h, z)
            step = solve(dF, F)
            z = z - step
            err = norm(step)
            E.append(err)
            if err <= 1e-12:
                break
        y = z

        if i < 3:
            [a, b] = np.polyfit(log(E[:-1]), log(E[1:]), 1)
            plt.figure(2+i)
            plt.clf()
            plt.loglog(E[:-1], exp(a*log(E[:-1]) + b))
            plt.loglog(E[:-1], E[1:], linestyle='None', marker='o')
            plt.legend()
            plt.title("t = {},  log e(n+1) ~ a log e(n),  a = {:.2f}".format(t, a))

        vy.append(y)
    return array(vy)


f = lambda t, y: array([y[0]*(1 - y[1]), y[1]*(-1.5 + 2*y[0])])
df = lambda t, y: array([[1 - y[1], -y[0]], [2*y[1], -1.5 + 2*y[0]]])
y0 = array([2, 2])

plt.figure(6)
plt.clf()
plt.title("Modèle de Lotka-Volterra")
plt.xlabel("Proies")
plt.ylabel("Prédateurs")

for h in [0.05, 0.005, 0.2]:
    vt = np.arange(0, 10+h, h)
    y = euler_imp(f, df, y0, vt)

    plt.figure(6)
    plt.plot(y[:,0], y[:,1], marker='.', markersize=2, linestyle='None',
             label="EI, h = {}".format(h))

plt.legend()




























