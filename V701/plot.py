import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon

count1, channel1, p1 = np.genfromtxt('data/werte1.txt', unpack=True)

efflng1 = 0.016*p1/1013

def f(x, A, B):
    return A*x + B

params, pcov = curve_fit(f, efflng1, count1)
errors = np.sqrt(np.diag(pcov))
print('Messung 1')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])


plt.plot(efflng1, count1, 'kx', label='Messwerte')
plt.plot(efflng1, f(count1, *params), 'b-', label='Fit')
plt.grid()
plt.legend()
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.savefig('build/plot1.pdf')
plt.clf()
