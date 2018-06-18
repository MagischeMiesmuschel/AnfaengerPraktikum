import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon

l1, t1 = np.genfromtxt('data/werte1.txt', unpack=True)
l2, t2 = np.genfromtxt('data/werte2.txt', unpack=True)
l3, I_0, I_x = np.genfromtxt('data/werte3.txt', unpack=True)

x1 = np.linspace(-5, 95, 20)
x2 = np.linspace(-5, 50, 20)
l1 = 2*l1

def f(x, A, B):
    return A*x + B

params, pcov = curve_fit(f, t1, l1)
errors = np.sqrt(np.diag(pcov))
print('Echo')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])
print('t =', params[1]/params[0]/2)
print('d =', params[1]/2)

plt.plot(t1, l1, 'kx', label='Messwerte')
plt.plot(x1, f(x1, *params), 'b-', label='Fit')
plt.grid()
plt.xlim(-5,95)
plt.ylim(-2,27)
plt.legend()
plt.xlabel(r'Laufzeit t in $\si{\micro\second}$')
plt.ylabel(r'doppelte Länge der Zylinder in cm')
plt.savefig('build/plot1.pdf')
plt.clf()


params, pcov = curve_fit(f, t2, l2)
errors = np.sqrt(np.diag(pcov))
print('Durchschallung')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])
print('t =', params[1]/params[0]/2)
print('d =', params[1]/2)

plt.plot(t2, l2, 'kx', label='Messwerte')
plt.plot(x2, f(x2, *params), 'b-', label='Fit')
plt.grid()
plt.xlim(-5,50)
plt.ylim(-2,13)
plt.legend()
plt.xlabel(r'Laufzeit t in $\si{\micro\second}$')
plt.ylabel(r'Länge der Zylinder in cm')
plt.savefig('build/plot2.pdf')
plt.clf()



params, pcov = curve_fit(f, l3, -np.log(I_x/I_0))
errors = np.sqrt(np.diag(pcov))
print('Daempfung')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])

plt.plot(l3, -np.log(I_x/I_0) , 'kx', label='Messwerte')
plt.plot(l3, f(l3, *params), 'b-', label='Fit')
plt.grid()
plt.legend()
plt.xlabel(r'Länge t in $\si{\centi\meter}$')
plt.ylabel(r'$-\ln(I(x)/I_0)$')
plt.savefig('build/plot3.pdf')
plt.clf()
