import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon

U1, I1 = np.genfromtxt('data/werteb.txt', unpack=True)
U2, I2 = np.genfromtxt('data/wertec.txt', unpack=True)
U3, I3 = np.genfromtxt('data/werted.txt', unpack=True)
U4, I4 = np.genfromtxt('data/werted2.txt', unpack=True)

def f(x, a, b):
    return a*x + b

# nomieren auf Ampere

I1 = I1*1e-3
I2 = I2*1e-3
I3 = I3*1e-3
I4 = I4*1e-3

# Fitten für b)

params, pcov = curve_fit(f, I1, U1)
errors = np.sqrt(np.diag(pcov))
print('Teil b)')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

R_i = -params[0]
U_0 = params[1]

# Plotten für b)

I = np.linspace(0, 75, 10)*1e-3
plt.plot(I1*1e3, U1, 'kx', label='Messwerte')
plt.plot(I*1e3, f(I, *params), 'r-', label='Fit')
plt.xlim(0, 75)
plt.xlabel(r'$I$ / mA')
plt.ylabel(r'$U_k$ / V')
plt.legend()
plt.grid()
plt.savefig('build/plot1.pdf')
plt.clf()

# Fitten für c)

params, pcov = curve_fit(f, I2, U2)
errors = np.sqrt(np.diag(pcov))
print('Teil c)')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

# Plotten für c)

I = np.linspace(0, 120, 10)*1e-3
plt.plot(I2*1e3, U2,'kx', label='Messwerte')
plt.plot(I*1e3, f(I, *params), 'r-', label='Fit')
plt.xlim(0, 120)
plt.xlabel(r'$I$ / mA')
plt.ylabel(r'$U_k$ / V')
plt.legend()
plt.grid()
plt.savefig('build/plot2.pdf')
plt.clf()

# Fitten für d)

params, pcov = curve_fit(f, I3, U3)
errors = np.sqrt(np.diag(pcov))
print('Teil d)')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

# Plotten für d)

I = np.linspace(0, 8, 10)*1e-3
plt.plot(I3*1e3, U3, 'kx', label='Messwerte')
plt.plot(I*1e3, f(I, *params), 'r-', label='Fit')
plt.xlim(0, 7.2)
plt.xlabel(r'$I$ / mA')
plt.ylabel(r'$U_k$ / V')
plt.legend()
plt.grid()
plt.savefig('build/plot3.pdf')
plt.clf()

# Fitten für d) 2

params, pcov = curve_fit(f, I4, U4)
errors = np.sqrt(np.diag(pcov))
print('Teil d) 2')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])

# Plotten für d) 2

I = np.linspace(0, 1.1, 10)*1e-3
plt.plot(I4*1e3, U4, 'kx', label='Messwerte')
plt.plot(I*1e3, f(I, *params), 'r-', label='Fit')
plt.xlim(0, 1.1)
plt.xlabel(r'$I$ / mA')
plt.ylabel(r'$U_k$ / V')
plt.legend()
plt.grid()
plt.savefig('build/plot4.pdf')
plt.clf()

# Plotten für e)

print(U_0)
print(R_i)

UdurchI = np.linspace(0, 55, 50)
plt.plot(U1/I1, U1*I1*1e3, 'kx', label='Messwerte')
plt.plot(UdurchI, (U_0**2/(UdurchI + R_i)**2)*(UdurchI)*1e3, label='Theoriekurve' )
plt.xlim(0, 55)
plt.xlabel(r'$R_a = \frac{U_k}{I} \,/\, \Omega$')
plt.ylabel(r'$N = U_k \cdot I \,/\, (10^{-3} \cdot W)$ ')
plt.legend()
plt.grid()
plt.savefig('build/plot5.pdf')
