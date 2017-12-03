import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math

def f(x, a, b, c):
    return a * (1/x**b) + c

def g(x, a, b, c):
    return a * np.cos(x + b) + c

Phase, U_out_2 = np.genfromtxt('data\Werte_2.txt', unpack=True) #Grad und V
Abstand_r, U_out_4 = np.genfromtxt('data\Werte_4.txt', unpack=True) #cm und mV
Phase = np.deg2rad(Phase)

params1, cov_matrix1 = curve_fit(g, Phase, U_out_2)
errors = np.sqrt(np.diag(cov_matrix1))
print('A1 = ', params1[0], '+/-', errors[0])
print('B1 = ', params1[1], '+/-', errors[1])
print('C1 = ', params1[2], '+/-', errors[2])

plt.plot(Phase, U_out_2, 'kx', label = 'Messwerte')
plt.plot(np.linspace(0,7), g(np.linspace(0,7), *params1), 'r-', label = 'Fit')
plt.plot(np.linspace(0,7), 10*2/math.pi * 0.3 * np.cos(np.linspace(0,7)), 'g--', label ='Theorie')
plt.grid()
plt.legend()
plt.xlabel('Phasenverschiebung in Bogenmaß')
plt.ylabel('Gleichspannung in V')
plt.xticks([0, (1/2)*math.pi, math.pi,(3/2)*math.pi, 2*math.pi],[0, '1/2 $\pi$', '$\pi$', '3/2 $\pi$', '2 $\pi$'])
plt.xlim(0, 2*math.pi)
plt.savefig('build\Plot1.pdf')
plt.clf()

params, cov_matrix = curve_fit(f, Abstand_r, U_out_4)
errors = np.sqrt(np.diag(cov_matrix))
print('A = ', params[0], '+/-', errors[0])
print('B = ', params[1], '+/-', errors[1])
print('C = ', params[2], '+/-', errors[2])

plt.plot(Abstand_r, U_out_4, 'kx', label = 'Messwerte')
plt.plot(Abstand_r, f(Abstand_r, *params), 'r-', label = 'Fit')
plt.grid()
plt.legend()
plt.xlabel('Abstand in cm')
plt.ylabel('Gleichspnnung in mV')
plt.xscale('log')
plt.yscale('log')
plt.xticks([5, 10, 15, 20, 25, 30])
plt.xlim(2.7, 30.7)
plt.savefig('build\Plot2.pdf')
plt.clf()