import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import scipy.constants as scicon

def f(x, a, b):
    return a*x + b

def B(I):
    return scicon.mu_0*(I*0.109**2)/(0.109**2+(0.138/2)**2)**(3/2)

r_kugel = 0.0255 # in m, Radius Kugel
l_stift = 0.012 # in m, Länge Stift
m_kugel = 0.1422 # in kg Masse Kugel

r_hebel, Ig = np.genfromtxt('werte1.txt', unpack=True)
Io, To = np.genfromtxt('werte2.txt', unpack=True)
Ik, Tk = np.genfromtxt('werte3.txt', unpack=True)

# Umwandeln in SI-Einheiten

r_hebel = r_hebel*1e-2

# Messfehler
T_error = np.ones(10)
T_error = 0.01*T_error
r_error = np.ones(10)
r_error = 1e-3*r_error
I_error = np.ones(10)
I_error = 0.1*I_error

print(r_error)
print(I_error)
print(T_error)

# Berechnung mit Gravitation
r = r_hebel + r_kugel + l_stift

count = range(10)
Bg = np.zeros(10)
B_error = np.zeros(10)

for x in count:
    Igu = ufloat(Ig[x],I_error[x])
    Bgu = B(Igu)
    Bg[x]= Bgu.n
    B_error[x] = Bgu.s

params,pcov = curve_fit(f, r, Bg)
print(params, np.sqrt(np.diag(pcov)), sep='\n')

plt.errorbar(r, Bg, xerr=r_error, yerr=B_error, fmt='k.', label='Messdaten')
plt.plot(r, f(r, *params), 'b-', label='Fit')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
