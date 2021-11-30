import numpy as np
import pylab as plt
import scipy
from numpy import array, pi, sin
from scipy.integrate import odeint

plt.ion()
plt.show()


### Exo 1

plt.figure(0, figsize=(16, 10))
plt.clf()

a = 0
b = 1
M = 299
h = (b-a)/(M+1)

x, h1 = np.linspace(a, b, M+2, retstep=True)
assert(h1 == h)
x1 = x[1:-1]

# bords = [0, 0]
bords = [2, -0.5]

u0 = lambda x: sin(pi*x) + sin(4*pi*x) + sin(20*pi*x)
u = u0(x)
y0 = u[1:-1]
# y0 = np.random.rand(M)

plt.axis([a, b, -1.0, 2.5])
plt.plot(x1, y0, marker='.')
plt.plot([a, b], bords, marker='.', linestyle="None", color='r')

A = (1/(h*h))*(-2*np.eye(M) + np.eye(M, k=1) + np.eye(M, k=-1))

tf = 0.01
dt = 0.00001
t = np.arange(0, tf+dt, dt)
assert(len(t) == int(round(tf/dt + 1)))

B = np.zeros(M)
B[0] = bords[0]/(h*h)
B[-1] = bords[1]/(h*h)

F = lambda y, t: A.dot(y) + B
Y = odeint(F, y0, t)

for i, y in enumerate(Y):
    plt.clf()
    plt.axis([a, b, -1.0, 2.5])
    plt.title("t = {:.4f}".format(i*dt))
    plt.plot(x1, y, marker='.')
    plt.plot([a, b], bords, marker='.', linestyle="None", color='r')
    plt.pause(0.02)

























