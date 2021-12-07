# WATANABE Izumi

import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy import array, sqrt, pi, cos, sin, log, exp
from scipy.integrate import odeint


## Question 5

u0 = array([-1, 0])
T = 6.8
h = 0.4
vt = np.arange(0, T+h, h)
size = 1.5

EE = array([[1, h], [-h, 1]]), "Euler explicite"
EI = 1/(1+h*h) * array([[1, h], [-h, 1]]), "Euler implicite"
ES = array([[1-h*h, h], [-h, 1]]), "Euler symplectique"
SV = np.eye(2) + h * array([[-h/2, 1], [-1+h*h/4, -h/2]]), "Störmer-Verlet"

plt.figure(0)
plt.clf()
plt.title("Oscillateur harmonique")
plt.xlabel("q(t)")
plt.ylabel("p(t)")
plt.axis('scaled')
plt.axis([-size, size, -size, size])

for A, label in [EE, EI, ES, SV]:

    u = u0
    vu = [u]
    for n in range(len(vt)-1):
        u = A.dot(u)
        vu.append(u)
    vu = array(vu)

    plt.plot(vu[:,0], vu[:,1], label=label)

plt.legend()


## Question 6

def stormer_verlet(f, u0, vt):
    u = u0
    vu = [u]
    h = vt[1] - vt[0]
    for n in range(len(vt)-1):
        [q, p] = u
        q1 = q + h*(p + h/2*f(q))
        p1 = p + h/2*(f(q) + f(q1))
        u = [q1, p1]
        vu.append(u)
    return array(vu)

norm2 = lambda q: sqrt(q[0]**2 + q[1]**2)
f = lambda q: -1/norm2(q)**3 * q
e = 0.6
T = 3*pi
h = 0.025
vt = np.arange(0, T+h, h)

u0 = array([[1-e, 0], [0, sqrt((1+e)/(1-e))]])
vu = stormer_verlet(f, u0, vt)

vtheta = np.linspace(0, 2*pi, 201)
r = lambda theta: sqrt((1 - e**2)/(1 - (e*cos(theta))**2))
vq1 = r(vtheta) * cos(vtheta) - e
vq2 = r(vtheta) * sin(vtheta)

plt.figure(1)
plt.clf()
plt.title("Kepler, q(t)")
plt.axis('scaled')
plt.axis([-2, 1, -1.5, 1.5])
plt.xlabel("q1(t)")
plt.ylabel("q2(t)")
plt.plot(vq1, vq2, label="Solution exacte")
plt.plot(vu[:, 0, 0], vu[:, 0, 1], label="Solution approchée",
    linestyle='None', marker='.', markersize=1)
plt.legend()

plt.figure(2)
plt.clf()
plt.title("Kepler, p(t)")
plt.axis('scaled')
plt.axis([-2, 2, -1, 3])
plt.xlabel("p1(t)")
plt.ylabel("p2(t)")
plt.plot(vu[:, 1, 0], vu[:, 1, 1], label="Solution approchée")
plt.legend()

vh = []
verr = []
for k in range(4, 15):
    h = 2**-k * pi
    vt = np.arange(0, T+h, h)
    vu = stormer_verlet(f, u0, vt)
    u = vu[-1, 0]
    err = norm2(u - array([-e-1, 0]))
    vh.append(h)
    verr.append(err)
[a, b] = np.polyfit(log(vh), log(verr), 1)

plt.figure(3)
plt.title("Erreur d'approximation de Kepler par Störmer-Verlet.  a = {:.3f}".format(a))
plt.xlabel("Pas h")
plt.ylabel("Erreur err")
plt.loglog(vh, exp(a*log(vh)+b), label="Régression linéaire, log(err) ~ a log(h)")
plt.loglog(vh, verr, linestyle='None', marker='o', label="Données")
plt.legend()
plt.show()
























