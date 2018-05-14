import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon

w_Au, R_Au = np.genfromtxt('data/Messwerte_Au.txt', unpack=True)
w_Br, R_Br = np.genfromtxt('data/Messwerte_Br.txt', unpack=True)
w_Sr, R_Sr = np.genfromtxt('data/Messwerte_Sr.txt', unpack=True)
w_Zn, R_Zn = np.genfromtxt('data/Messwerte_Zn.txt', unpack=True)
w_Zr, R_Zr = np.genfromtxt('data/Messwerte_Zr.txt', unpack=True)
w_Bragg, R_Bragg = np.genfromtxt('data/Messwerte1_Bragg.txt', unpack=True)
w_Cu, R_Cu = np.genfromtxt('data/Messwerte2.txt', unpack=True)

w_Au = w_Au/2
w_Br = w_Br/2
w_Sr = w_Sr/2
w_Zn = w_Zn/2
w_Zr = w_Zr/2
w_Bragg = w_Bragg/2
w_Cu = w_Cu/2

w1 = w_Cu[79:83]
R1 = R_Cu[79:83]
w2 = w_Cu[90:95]
R2 = R_Cu[90:95]

plt.plot(w_Cu, R_Cu, 'b-', label='Bremsberg')
plt.plot(w2, R2, 'g-', label=r'$K_\alpha$')
plt.plot(w1, R1, 'r-', label=r'$K_\beta$')
plt.grid()
plt.ylim(0, 3000)
plt.legend()
plt.xlabel('Kristallwinkel $ \Theta$ in $^\circ$')
plt.ylabel(r'Zählrate in Impulse/$s$')
plt.savefig('build/plot2.pdf')
plt.clf()



plt.plot(w_Bragg, R_Bragg, 'b-', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel('Kristallwinkel $ \Theta$ in $^\circ$')
plt.ylabel(r'Zählrate in Impulse/$s$')
plt.savefig('build/plot1.pdf')
plt.clf()

plt.plot(w_Zn, R_Zn, 'b-', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel('Kristallwinkel $ \Theta$ in $^\circ$')
plt.ylabel(r'Zählrate in Impulse/$s$')
plt.savefig('build/plot3.pdf')
plt.clf()

plt.plot(w_Br, R_Br, 'b-', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel('Kristallwinkel $ \Theta$ in $^\circ$')
plt.ylabel(r'Zählrate in Impulse/$s$')
plt.savefig('build/plot4.pdf')
plt.clf()

plt.plot(w_Sr, R_Sr, 'b-', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel('Kristallwinkel $ \Theta$ in $^\circ$')
plt.ylabel(r'Zählrate in Impulse/$s$')
plt.savefig('build/plot5.pdf')
plt.clf()

plt.plot(w_Zr, R_Zr, 'b-', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel('Kristallwinkel $ \Theta$ in $^\circ$')
plt.ylabel(r'Zählrate in Impulse/$s$')
plt.savefig('build/plot6.pdf')
plt.clf()

plt.plot(w_Au, R_Au, 'b-', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel('Kristallwinkel $ \Theta$ in $^\circ$')
plt.ylabel(r'Zählrate in Impulse/$s$')
plt.savefig('build/plot8.pdf')
plt.clf()




def g(x, A, B):
    return A*x + B

x = np.array([30, 35, 38, 40])
y = np.array([9.65, 13.48, 16.12, 18.01])

y = y*1000*1.6*10**(-19)
y = np.sqrt(y)

params, pcov = curve_fit(g, x, y)
errors = np.sqrt(np.diag(pcov))
print('Raumladung')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])
plt.plot(x, y, 'kx', label='Messwerte')
plt.plot(x, g(x, *params), 'b-', label='Fit')
plt.grid()
plt.legend()
plt.ylabel(r'$\sqrt{E_\text{abs}}$ in $\sqrt{J}$')
plt.xlabel(r'$Z$')
plt.savefig('build/plot7.pdf')
plt.clf()
