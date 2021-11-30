import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.integrate import odeint
from scipy.optimize import fsolve
from numpy import array, exp, log, sqrt, cos, sin, pi

plt.ion()
plt.show()


## Exercice 1

f = lambda x: x - sin(x) - 1.5*pi
df = lambda x: 1 - cos(x)
X = np.linspace(0, 10, 501)

plt.figure(0)
plt.clf()
plt.axhline(color='black', linewidth=1)
plt.plot(X, f(X), label="f(x) = x - sin(x) - 1.5π")
plt.legend()

s = fsolve(f, 4)[0]
plt.plot(s, 0, marker='|', markersize=15)
plt.title("s = {:.6f}".format(s))

x = 2
E, vx = [], [x]

for n in range(1, 20):
    x1 = x - f(x)/df(x)
    vx.append(x)
    err = np.abs(x - x1)
    E.append(err)
    print("{:3d} {:25.20f} {:25.20f}".format(n, err, np.abs(x1 - s)))
    x = x1
    if err <= 1e-8:
        break

[a, b] = np.polyfit(log(E[:-1]), log(E[1:]), 1)

plt.figure(1)
plt.clf()
plt.loglog(E[:-1], exp(a*log(E[:-1]) + b))
plt.loglog(E[:-1], E[1:], linestyle='None', marker='o')
plt.legend()
plt.title("log e(n+1) ~ a log e(n),  a = {:.2f}".format(a))

plt.figure(0)
p = len(vx)
for i in range(p):
    x = vx[i]
    plt.plot([x, x], [0, f(x)], color='g')
    if i < p-1:
        plt.plot([x, vx[i+1]], [f(x), 0], color='g')





























