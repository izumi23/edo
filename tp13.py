import numpy as np
import scipy.interpolate, scipy.linalg
import matplotlib.pyplot as plt
from numpy import array, cos, sin, pi, exp, log, log10, abs, max
from scipy.integrate import quad, odeint
from scipy.linalg import norm, solve

plt.ion()
plt.show()


## Question 1

f = lambda t: exp(t)/log(t)
df = lambda t: exp(t)/log(t)**2 * (log(t) - 1/t)
a, b = 2, 4
vm = []
vI = []

for k in range(1, 16):
    m = 2**k
    vt = np.linspace(a, b, m+1)
    h = 2**(1-k)
    I = 0
    for i in range(m):
        I += 2*f(vt[i]) + 4*f(vt[i+1]) - h*df(vt[i+1])
    vm.append(m)
    vI.append(h/6 * I)

I_exact = quad(f, a, b)[0]

vm = array(vm)
verr = abs(array(vI) - I_exact)

[p1, p0] = np.polyfit(log10(vm), log10(verr), 1)

plt.figure(0)
plt.clf()
plt.xlabel("m")
plt.ylabel("|I - I_m|")
plt.loglog(vm, 10**(p1*log10(vm) + p0), label="log |I - I_m| ~ a log m")
plt.loglog(vm, verr, linestyle='None', marker='o', label="Données")
plt.legend()
plt.title("Approximation de l'intégrale, a = {:.2f}".format(p1))


## Question 2

def enright(f, df, y0, vt):
    # y = np.asarray(y0).ravel()
    y = y0
    d = len(y)
    h = vt[1] - vt[0]
    vy = [y]
    for i in range(len(vt)-1):
        z = y
        err = 1
        while norm(err) > 1e-8:
            M = 2/3*h*df(z) - 1/6*h**2* df(y + h*f(y)) @ df(z) - np.eye(d)
            N = y + 1/3*h*f(y) + 2/3*h*f(z) - 1/6*h**2*df(y + h*f(y)) @ f(z) - z
            err = solve(M, N)
            z = z - err
        y = z
        vy.append(y)
    return array(vy)


## Question 3

f = lambda x: x*(1 - x)
df = lambda x: 1 - 2*x

y0 = array([0.1])
y_exact = np.vectorize(lambda t: y0 / (y0 - exp(-t)*(y0 - 1)))
t0, tf = 0, 2

vh = []
verr = []
for k in range(3, 13):
    h = 2**-k
    vt = np.arange(t0, tf+h, h)
    vy = enright(f, df, y0, vt)[:,0]
    vy_exact = y_exact(vt)
    err = max(abs(vy - vy_exact))
    vh.append(h)
    verr.append(err)

vh, verr = array(vh), array(verr)
[p1, p0] = np.polyfit(log10(vh), log10(verr), 1)

plt.figure(1)
plt.clf()
plt.xlabel("h")
plt.ylabel("|I - I_h|")
plt.loglog(vh, 10**(p1*log10(vh) + p0), label="log |I - I_h| ~ a log h")
plt.loglog(vh, verr, linestyle='None', marker='o', label="Données")
plt.legend()
plt.title("Modèle logistique avec Enright, a = {:.2f}".format(p1))


## Question 4

a = 30
f = lambda x: array([x[1], -x[0] + a*(1 - x[0]**2)*x[1]])
df = lambda x: array([[0, 1], [-1 + -2*a*x[0]*x[1], a*(1 - x[0]**2)]])

y0 = array([1, 0])
t0, tf = 0, 5*a
vy_exact = odeint(lambda y,t: f(y), y0, np.arange(t0, tf+0.001, 0.001))
vy = enright(f, df, y0, np.arange(t0, tf+0.01, 0.01))

plt.figure(2)
plt.clf()
plt.xlabel("theta(t)")
plt.ylabel("theta'(t)")
plt.plot(vy_exact[:,0], vy_exact[:,1], label="odeint")
plt.plot(vy[:,0], vy[:,1], label="enright", linestyle='None', marker='o', markersize=2)
plt.legend()
plt.title("Van der Pol avec Enright")

