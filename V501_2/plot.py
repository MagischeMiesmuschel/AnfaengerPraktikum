import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math

U_d1, U_d2, U_d3, U_d4, U_d5, d = np.genfromtxt('data/werte1_a.txt', unpack=True)
U_Ba1 = 200 #Volt
U_Ba2 = 230
U_Ba3 = 250
U_Ba4 = 280
U_Ba1 = 300
nu, n = np.genfromtxt('data/werte1_b.txt', unpack=True) #Herz
nu_neu = nu*n
y_ausl = 0.013  #Meter

I1, I2 = np.genfromtxt('data/werte2_a.txt', unpack=True) #Amper

print("U_d1: ", U_d1)
print("U_d2: ", U_d2)
print("U_d3: ", U_d3)
print("U_d4: ", U_d4)
print("U_d5: ", U_d5)
print("nu: ", nu)
print("I1: ", I1)
print("I2: ", I2)

nu_mittel = (nu[0]*n[0] + nu[1]*n[1] + nu[2]*n[2] + nu[3]*n[3]) / 4

print("nu_mittel: ", nu_mittel)
print("nu_neu: ", nu_neu)

nu_abw = ((nu_mittel - nu_neu[0]) + (nu_neu[1] - nu_mittel) + (nu_neu[2] - nu_mittel) + (nu_mittel - nu_neu[3]))/3

print("nu_abw: ", nu_abw)

plt.plot(U_d1, d)
plt.savefig('build/plot1.pdf')