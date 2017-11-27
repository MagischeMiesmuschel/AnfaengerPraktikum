import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import matplotlib.ticker

n1, U1 = np.genfromtxt('werte1.txt', unpack=True) #Rechteck
n2, U2 = np.genfromtxt('werte2.txt', unpack=True) #Sägezahn
n3, U3 = np.genfromtxt('werte3.txt', unpack=True) #Dreieck

frequenz1 = 120e3 # Hertz
frequenz2 = 100e3 # Hertz
frequenz3 = 100e3 # Hertz

def f(x, a, b):
    return a*x + b

# nomieren auf die erste Amplitude

U1 = U1/U1[0]
U2 = U2/U2[0]
U3 = U3/U2[0]

#doppellogarithmische Darstellung

U1_log = np.log(U1)
U2_log = np.log(U2)
U3_log = np.log(U3)

n1_log = np.log(n1)
n2_log = np.log(n2)
n3_log = np.log(n3)

# Fitten für Rechteck

params, pcov = curve_fit(f,n1_log,U1_log)
errors = np.sqrt(np.diag(pcov))
print('Rechteck:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

#Plotten für Rechteck

fig1, ax1 = plt.subplots()
ax1.plot(n1, U1, 'kx', label='Messwerte')
ax1.plot(n1, np.exp(f(n1_log, *params)), 'r-', label='Fit')
ax1.grid()
ax1.legend()
ax1.set_xlabel(r"Nummer der Oberwelle")
ax1.set_ylabel(r"$\frac{U}{U_0}$", rotation=0)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig1.tight_layout()
plt.savefig('build/plot1.pdf')
plt.clf()

# Fitten für Sägezahn

params, pcov = curve_fit(f,n2_log,U2_log)
errors = np.sqrt(np.diag(pcov))
print('Sägezahn:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

#Plotten für Sägezahn

fig2, ax2 = plt.subplots()
ax2.plot(n2, U2, 'kx', label='Messwerte')
ax2.plot(n2, np.exp(f(n2_log, *params)), 'r-', label='Fit')
ax2.grid()
ax2.legend()
ax2.set_xlabel(r"Nummer der Oberwelle")
ax2.set_ylabel(r"$\frac{U}{U_0}$", rotation=0)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xticks([1, 3, 5, 7, 9])
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig2.tight_layout()
plt.savefig('build/plot2.pdf')
plt.clf()

# Fitten für Dreieck

params, pcov = curve_fit(f,n3_log,U3_log)
errors = np.sqrt(np.diag(pcov))
print('Dreieck:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

#Plotten für Dreieck

fig3, ax3 = plt.subplots()
ax3.plot(n3, U3, 'kx', label='Messwerte')
ax3.plot(n3, np.exp(f(n3_log, *params)), 'r-', label='Fit')
ax3.grid()
ax3.legend()
ax3.set_xlabel(r"Nummer der Oberwelle")
ax3.set_ylabel(r"$\frac{U}{U_0}$", rotation=0)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17])
ax3.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig3.tight_layout()
plt.savefig('build/plot3.pdf')
plt.clf()
