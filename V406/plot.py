import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values

def theory(zeta, A0, zeta_0, b):
    return (A0 * b * np.sinc(b * np.sin(zeta - zeta_0) / 635e-9))**2

def theory2(zeta, A0, zeta_0, b, d):
    return (2 * np.cos(np.pi*d*np.sin(zeta - zeta_0) / 635e-9))**2*theory(zeta, A0, zeta_0, b)

I_dunkel = 0.85e-9 #Dunkelstrom
b = 0.15e-3 #Breite pro Spalt
L = 1 #Abstand von Spalt zu Detektor

I_doppel1, x_doppel1 = np.genfromtxt("data/werte1.txt", unpack=True) #Doppelspalt1
print('Tabelle Doppel1:')
for i in range(30):
    print(x_doppel1[i], " & " ,I_doppel1[i]," & ",x_doppel1[i+30], " & " ,I_doppel1[i+30], "\\\\")
x_doppel1 *= 1e-3 #s/m
I_doppel1 *= 1e-6 #I/A
a1 = 0.25e-3 #Abstand 1 der Spalte
I_doppel1 = I_doppel1 - I_dunkel

x_doppel2, I_doppel2 = np.genfromtxt("data/werte2.txt", unpack=True) #Doppelspalt2
print('Tabelle Doppel2:')
for i in range(23):
    print(x_doppel2[i], " & " ,I_doppel2[i]," & " ,x_doppel2[i+23], " & " ,I_doppel2[i+23], "\\\\")
x_doppel2 *= 1e-3 #s/m
I_doppel2 *= 1e-6 #I/A
a2 = 0.35e-3 #Abstand 1 der Spalte
I_doppel2 = I_doppel2 - I_dunkel

x_einzel, I_einzel = np.genfromtxt("data/werte3.txt", unpack=True) #Einzelspalt
print('Tabelle Einzelspalt:')
for i in range(21):
    print(x_einzel[i], " & " ,I_einzel[i]," & " ,x_einzel[i+21], " & " ,I_einzel[i+21], "\\\\")
x_einzel *= 1e-3 #s/m
I_einzel *= 1e-6 #I/A
I_einzel = I_einzel -I_dunkel

x_fit1 = np.linspace(0.009, 0.033, 1000)
x_fit2 = np.linspace(0.015, 0.031, 1000)
x_fit3 = np.linspace(0.019, 0.0264, 1000)



############################################################

params1, covariance_matrix1 = optimize.curve_fit(theory, x_einzel, I_einzel, p0=[10, 0.02, 1e-3])

A_0, x_0, b = correlated_values(params1, covariance_matrix1)

print('Jetzt kommt der erste Einzelspalt: ')
print('A_0 =', A_0)
print('x_0 =', x_0)
print('b =', b)

plt.plot(x_fit1*1e3, theory(x_fit1, *params1)*1e6, 'b', label="Fit")
plt.plot(x_einzel*1e3, I_einzel*1e6, 'rx', label="Messwerte")
plt.grid()
plt.legend()
plt.xlim(9, 33)
plt.xlabel(r'Position in mm')
plt.ylabel(r'Intensität in µA')
plt.savefig("build/plot1.pdf")
plt.clf()

############################################################

params2, covariance_matrix2 = optimize.curve_fit(theory2, x_doppel1, I_doppel1, p0=[10, 0.0231, 0.15e-3, 0.25e-3])

A_0, x_0, b, d = correlated_values(params2, covariance_matrix2)

print('Jetzt kommt der Doppelspalt 1: ')
print('A_0 =', A_0)
print('x_0 =', x_0)
print('b =', b)
print('d =', d)

plt.plot(x_fit2*1e3, theory2(x_fit2, *params2)*1e6, 'b', label="Fit")
plt.plot(x_doppel1*1e3, I_doppel1*1e6, 'rx', label="Messwerte")
plt.grid()
plt.legend()
plt.xlim(15, 31)
plt.xlabel(r'Position in mm')
plt.ylabel(r'Intensität in µA')
plt.savefig("build/plot2.pdf")
plt.clf()

############################################################

params3, covariance_matrix3 = optimize.curve_fit(theory2, x_doppel2, I_doppel2, p0=[10, 0.0225, 0.15e-3, 0.35e-3])

A_0, x_0, b, d = correlated_values(params3, covariance_matrix3)

print('Jetzt kommt der Doppelspalt 2: ')
print('A_0 =', A_0)
print('x_0 =', x_0)
print('b =', b)
print('d =', d)

plt.plot(x_fit3*1e3, theory2(x_fit3, *params3)*1e6, 'b', label="Fit")
plt.plot(x_doppel2*1e3, I_doppel2*1e6, 'rx', label="Messwerte")
plt.grid()
plt.legend()
plt.xlim(19, 26.5)
plt.xlabel(r'Position in mm')
plt.ylabel(r'Intensität in µA')
plt.savefig("build/plot3.pdf")
plt.clf()

############################################################

plt.plot(x_fit1*1e3, theory(x_fit1, *params1)*1e6, 'b', label="Fit Einzelspalt")
plt.plot(x_einzel*1e3, I_einzel*1e6, 'rx', label="Messwerte Einzelspalt")
plt.grid()
plt.xlim(9, 33)
plt.xlabel(r'Position in mm')
plt.ylabel(r'Intensität in µA')
plt.plot(x_fit2*1e3-2.13, theory2(x_fit2, *params2)*1e6*1.092, 'k', label="Fit Doppelspalt")
plt.plot(x_doppel1*1e3-2.13, I_doppel1*1e6*1.092, 'gx', label="Messwerte Doppelspalt")
plt.legend()
plt.savefig("build/plot4.pdf")