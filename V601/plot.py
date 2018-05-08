import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
from scipy.stats import sem

k1, k2 = np.genfromtxt('data/werte1.txt', unpack=True)
x1, y1 = np.genfromtxt('data/werte2.txt', unpack=True)
x2, y2 = np.genfromtxt('data/werte3.txt', unpack=True)
k3 = np.genfromtxt('data/werte4.txt', unpack=True)
oz1, ab1 = np.genfromtxt('data/werte5.txt', unpack=True)

k1 = 2/k1
k2 = 2/k2
k3 = 10/k3

print("Skalierung")
S1 = np.mean(k1)
S1err = sem(k1)
print(S1, S1err)
S2 = np.mean(k2)
S2err = sem(k2)
print(S2, S2err)
S3 = np.mean(k3)
S3err = sem(k3)
print(S3, S3err)
print()

Stelle1 = x1
Stelle2 = x2
x1 = x1*S1
x2 = x2*S2
y1 = y1/10
y1[19:30] = y1[19:30]*2
y2 = y2/5

gx = 230* np.ones(15)*S1
gy = np.linspace(-0.2, 5, 15)

plt.plot(x1, y1, 'bx', label='Messwerte')
plt.plot(gx, gy, 'k--')
plt.ylim(-0.2, 5)
plt.grid()
plt.legend()
plt.xlabel(r'Bremsspannung $U_\text{a}$ in V')
plt.ylabel(r'Steigung')
plt.savefig('build/plot1.pdf')
plt.clf()

plt.plot(x2, y2, 'bx', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel(r'Bremsspannung $U_\text{a}$ in V')
plt.ylabel(r'Steigung')
plt.savefig('build/plot2.pdf')
plt.clf()

print("Tabelle")
i = 0
while i < 30:
    print(Stelle1[i], x1[i], y1[i], Stelle2[i], x2[i], y2[i])
    i += 1
print()
i = 0
while i < 36:
    print(Stelle2[i], x2[i], y2[i])
    i +=1
print()

print("Arbeit")
i = 0
while i < 7:
    print(oz1[i],ab1[i], ab1[i]*S3)
    i +=1
print()
ab1 = ab1*S3
print(np.mean(ab1), sem(ab1))

#test
