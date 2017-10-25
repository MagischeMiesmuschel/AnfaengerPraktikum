import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

def f(x, a, b):
    return a*x + b

def B(I):
    return (I*0.109^2)/(0.109^2+(0.138/2)^2)^(3/2)

r_kugel = 0.0255 # in m, Radius Kugel
l_stift = 0.012 # in m, Länge Stift
m_kugel = 0.1422 # in kg Masse Kugel

r_hebel, Ig = np.genfromtxt('werte1.txt', unpack=True)
Io, To = np.genfromtxt('werte2.txt', unpack=True)
Ik, Tk = np.genfromtxt('werte3.txt', unpack=True)

# Berechnung mit Gravitation

r = ufloat(r_hebel + r_kugel + l_stift,)


# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot.pdf')
