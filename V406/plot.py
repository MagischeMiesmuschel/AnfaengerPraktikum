import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math
from uncertainties import ufloat
from uncertainties import correlated_values
from scipy import optimize

def theory(zeta, A0, zeta_0, b):
    return (A0 * b * np.sinc(b * np.sin((zeta - zeta_0)/(1) / 635e-9))**2)

def einzel(x, A, B, x0):
    return A * (np.sin((math.pi / (635e-9)) * B * np.sin(x-x0))/((math.pi / (635e-9)) * B * np.sin(x-x0)))**2

def f(x,a,b):
    return a**2 * b * (635e-9 / (math.pi * b * np.sin(x)))**2 * (np.sin((math.pi * b * np.sin(x))/ 635e-9))**2

def g1(x,a,b):
    return a**2 * 2 * (np.cos((math.pi * 0.25e-3 * np.sin(x)) / 635e-9))**2 * (635e-9 / (math.pi * b * np.sin(x)))**2 * (np.sin((math.pi * b * np.sin(x))/ 635e-9))**2

def g2(x,a,b):
    return a**2 * 2 * (np.cos((math.pi * 0.35e-3 * np.sin(x)) / 635e-9))**2 * (635e-9 / (math.pi * b * np.sin(x)))**2 * (np.sin((math.pi * b * np.sin(x))/ 635e-9))**2

I_dunkel = 0.85e-9 #Dunkelstrom
b = 0.15e-3 #Breite pro Spalt
L = 1 #Abstand von Spalt zu Detektor

I_doppel1, x_doppel1 = np.genfromtxt("data/werte1.txt", unpack=True) #Doppelspalt1
a1 = 0.25e-3 #Abstand 1 der Spalte
#I_doppel1 = I_doppel1 - I_dunkel

x_doppel2, I_doppel2 = np.genfromtxt("data/werte2.txt", unpack=True) #Doppelspalt2
a2 = 0.35e-3 #Abstand 1 der Spalte
#I_doppel1 = I_doppel2 - I_dunkel

x_einzel, I_einzel = np.genfromtxt("data/werte3.txt", unpack=True) #Einzelspalt
#I_einzel = I_einzel -I_dunkel

phi_einzel = x_einzel - 20.99   #Berechnung von Phi mit x_0 wo das Max ist
phi_doppel1 = x_doppel1 - 23.1
phi_doppel2 = x_doppel2 - 22.75

x = np.linspace(-11e-3,11e-3,1000)

x_einzel *= 1e-3
I_einzel *= 1e-6

params1, pcov1 = curve_fit(einzel,x_einzel,I_einzel)

print("A0 Einzelspalt: ", params1[0], "+/-", pcov1[0][0])
print("Spaltbreite Einzelspalt: ", params1[1], "+/-", pcov1[1][1])
print("x0 Einzelspalt: ", params1[2], "+/-", pcov1[2][2])


plt.plot(x, einzel(x, *params1), 'b', label="Fit")

plt.plot(x_einzel-21e-3, I_einzel, 'rx', label="Messwerte")
plt.grid()
plt.legend()
plt.xlabel("Position in mm")
plt.ylabel("Intensität in µA")
plt.savefig("build/plot1.pdf")
plt.clf()

params2, pcov2 = curve_fit(g1,phi_doppel1,I_doppel1)

print("A0 Doppelspalt1: ", params2[0], "+/-", pcov2[0][0])
print("Spaltbreite Einzelspalt: ", params2[1], "+/-", pcov2[1][1])

plt.plot(phi_doppel1, g1(phi_doppel1, *params2), 'b', label="Fit")

plt.plot(phi_doppel1, I_doppel1, 'rx', label="Messwerte")
plt.grid()
plt.legend()
plt.xlabel("Position in mm")
plt.ylabel("Intensität in µA")
plt.savefig("build/plot2.pdf")
plt.clf()

params3, pcov3 = curve_fit(g2,phi_doppel2,I_doppel2)

print("A0 Doppelspalt2: ", params3[0], "+/-", pcov3[0][0])
print("Spaltbreite Einzelspalt: ", params3[1], "+/-", pcov3[1][1])

plt.plot(phi_doppel2, g2(phi_doppel2, *params3), 'b', label="Fit")

plt.plot(phi_doppel2, I_doppel2, 'rx', label="Messwerte")
plt.grid()
plt.legend()
plt.xlabel("Position in mm")
plt.ylabel("Intensität in µA")
plt.savefig("build/plot3.pdf")
plt.clf()