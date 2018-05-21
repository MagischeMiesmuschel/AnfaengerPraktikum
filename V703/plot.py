import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy
import scipy.constants as scicon
import math

def f(x, m, n):
    return m*x+n

U,N,I = np.genfromtxt("data/werte.txt", unpack=True)
N_1 = 36318
N_2 = 30562
N_12= 65063
t = 60
sq_N = N**(0.5)

sq_N_1 = N_1**(0.5)
N_1 = ufloat(N_1,sq_N_1)

sq_N_2 = N_2**(0.5)
N_2 = ufloat(N_2,sq_N_2)

sq_N_12 = N_12**(0.5)
N_12 = ufloat(N_12,sq_N_12)

N_neu = unumpy.uarray(N,sq_N)

print(sq_N)
print("N_neu: ", N_neu)

#Chrakteristika

x = np.linspace(310,700, 1000)
nichts, N_fit, nichts2 = np.genfromtxt("data/hilf.txt", unpack=True)
params1, pcov1 = curve_fit(f, nichts, N_fit)
print("Steigung: ", params1[0])

#plt.plot(U,N,'kx',label="Charakteristika")
plt.plot(x, f(x, *params1), 'r', label="Fit")
plt.errorbar(U, N, yerr=sq_N, fmt='k.', label="Messwerte")
plt.grid()
plt.legend()
plt.xlabel("U in V")
plt.ylabel("N")
plt.savefig("build/plot1.pdf")
plt.axis([310, 710, 15000, 19000])
plt.savefig("build/plot2.pdf")
plt.clf()

#Totzeitberechnung

T = (N_1 + N_2 - N_12)/(2 * N_1 * N_2)
print("Totzeit: ", T)

#Ladungsmenge

Q = (I * 10**(-6) * t) / N
print("Q: ", Q)
Q_ele = Q / (1.602*(10**(-19)))
print("Q_ele: ", Q_ele)

Q_del = (I * 10**(-6) *t)/(N**2)*(sq_N)

print("Q_del: ", Q_del)

Q_ele_del = 1 / (1.602*(10**(-19))) * (Q_del)

print("Q_ele_del: ", Q_ele_del)