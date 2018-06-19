import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon

count1, channel1, p1 = np.genfromtxt('data/werte1.txt', unpack=True)
count2, channel2, p2 = np.genfromtxt('data/werte2.txt', unpack=True)
verteilung = np.genfromtxt('data/werte3.txt', unpack=True)

efflng1 = 0.016*p1/1013
efflng2 = 0.022*p2/1013
e1 = channel1*(4/channel1[0])
e2 = channel2*(4/channel2[0])

def f(x, A, B):
    return A*x + B

kritx1 = efflng1[7:16]
krity1 = count1[7:16]
x_fit1 = np.linspace(-0.001, 0.016, 20)

params, pcov = curve_fit(f, kritx1, krity1)
errors = np.sqrt(np.diag(pcov))
print('Messung 1')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])
A = ufloat(params[0], errors[0])
B = ufloat(params[1], errors[1])
print('R =', (135236/2-B)/A)

plt.plot(efflng1, count1, 'kx', label='Messwerte')
plt.plot(kritx1, krity1, 'cx')
plt.plot(x_fit1, f(x_fit1, *params), 'b-', label='Fit')
plt.grid()
plt.xlim(-0.001, 0.016)
plt.legend()
plt.xlabel(r'effektive Länge in $\si{\meter}$')
plt.ylabel(r'Zählrate')
plt.savefig('build/plot1.pdf')
plt.clf()



kritx2 = efflng2[7:15]
krity2 = count2[7:15]
x_fit2 = np.linspace(-0.0005, 0.0175, 20)

params, pcov = curve_fit(f, kritx2, krity2)
errors = np.sqrt(np.diag(pcov))
print('Messung 2')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])
A = ufloat(params[0], errors[0])
B = ufloat(params[1], errors[1])
print('R =', (92045/2-B)/A)

plt.plot(efflng2, count2, 'kx', label='Messwerte')
plt.plot(kritx2, krity2, 'cx')
plt.plot(x_fit2, f(x_fit2, *params), 'b-', label='Fit')
plt.grid()
plt.xlim(-0.0005, 0.0175)
plt.legend()
plt.xlabel(r'effektive Länge in $\si{\meter}$')
plt.ylabel(r'Zählrate')
plt.savefig('build/plot2.pdf')
plt.clf()



params, pcov = curve_fit(f, efflng1, e1)
errors = np.sqrt(np.diag(pcov))
print('Messung 1 Energie')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])

plt.plot(efflng1, e1, 'kx', label='Messwerte')
plt.plot(efflng1, f(efflng1, *params), 'b-', label='Fit')
plt.grid()
plt.legend()
plt.xlabel(r'effektive Länge in $\si{\meter}$')
plt.ylabel(r'Energie in MeV')
plt.savefig('build/plot3.pdf')
plt.clf()



params, pcov = curve_fit(f, efflng2, e2)
errors = np.sqrt(np.diag(pcov))
print('Messung 2 Energie')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])

plt.plot(efflng2, e2, 'kx', label='Messwerte')
plt.plot(efflng2, f(efflng2, *params), 'b-', label='Fit')
plt.grid()
plt.legend()
plt.xlabel(r'effektive Länge in $\si{\meter}$')
plt.ylabel(r'Energie in MeV')
plt.savefig('build/plot4.pdf')
plt.clf()

#

print('Verteilung')
histogram = plt.figure()

print(np.mean(verteilung))
print(np.std(verteilung))
gaussian_numbers = np.random.normal(85, 12, 10000)
poisson_numbers = np.random.poisson(85, 10000)

plt.hist([gaussian_numbers, verteilung, poisson_numbers], 20, label=['Gaußverteilung', 'Messwerte', 'Poissonverteilung'], normed=1)
plt.legend()
plt.xlabel(r'Zählrate')
plt.ylabel(r'Häufigkeit')
plt.savefig('build/plot5.pdf')
plt.clf()
