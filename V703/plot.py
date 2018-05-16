import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
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
sq_N_2 = N_2**(0.5)
sq_N_12 = N_12**(0.5)

print(sq_N)

#Chrakteristika

x = np.linspace(310,700, 1000)
nichts, N_fit, nichts2 = np.genfromtxt("data/hilf.txt", unpack=True)
params1, pcov1 = curve_fit(f, U, N)
print("Steigung: ", params1[0])

plt.plot(U,N,'kx',label="Charakteristika")
plt.plot(x, f(x, *params1), 'r', label="Fit")
plt.grid()
plt.legend()
plt.xlabel("U in V")
plt.ylabel("N")
plt.savefig("build/plot1.pdf")
plt.axis([310, 710, -500, 22500])
plt.clf()

#Totzeitberechnung

T = (N_1 + N_2 - N_12)/(2 * N_1 * N_2)