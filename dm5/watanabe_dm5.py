# WATANABE Izumi

import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import array, sqrt, pi, cos, sin, log2, exp
from scipy.integrate import odeint
from scipy.linalg import norm, solve

a = -1/12
b = 2/3
c = 5/12

def predcor(f, df, y0, vt, printnewton=False):

    d = len(y0)
    h = vt[1] - vt[0]
    t = vt[0]
    yold = y0
    p1 = f(t, y0)
    p2 = f(t+h, y0 + h*p1)
    y = y0 + h/2*(p1 + p2)
    vy = [y0, y]

    for n in range(1, len(vt)-1):
        t = vt[n]
        yold = y
        z = y
        E = []
        pr = printnewton and n <= 3
        if pr: print("Pas", n)
        for k in range(1, 11):
            F = z - y - h*(a*f(t+h, z) + b*f(t, y) + c*f(t-h, yold))
            dF = np.eye(d) - h*a*df(t+h, z)
            step = solve(dF, F)
            z = z - step
            err = norm(step)
            E.append(err)
            if pr: print(k, err)
            if err <= 1e-12:
                if pr: print()
                break
        y = z

        vy.append(y)
    return array(vy)

alpha = 3
f = lambda t, y: array([y[1], -y[0] + alpha*(1 - y[0]**2)*y[1]])

df = lambda t, y: \
    array([[0, 1], [-1 - 2*alpha*y[0]*y[1], alpha*(1 - y[0]**2)]])

fsym = lambda y, t: f(t, y)

y0 = array([3., 0.])
t0, tf = 0, 5*alpha
ysol = odeint(fsym, y0, np.arange(t0, tf+0.001, 0.001))

plt.figure(0)
plt.title("Oscillateur de Van der Pol")
plt.xlabel("θ(t)")
plt.ylabel("θ'(t)")

plt.plot(ysol[:,0], ysol[:,1], label="odeint")
for h in [0.01, 0.001]:
    vt = np.arange(t0, tf+h, h)
    vy = predcor(f, df, y0, vt)
    plt.plot(vy[:,0], vy[:,1], label="predcor, h = {}".format(h))

vh, vyf = [], []
for k in range(4, 11):
    h = 2**-k
    vt = np.arange(t0, tf+h, h)
    vy = predcor(f, df, y0, vt)
    vh.append(h)
    vyf.append(vy[-1])

vh = array(vh)
vyf = array(vyf)

dyf = array([norm(z, 2) for z in vyf[:-1] - vyf[1:]])
order = log2(dyf[:-1]) - log2(dyf[1:])
print(vyf, dyf, order, sep='\n')

plt.legend()
plt.show()


































