import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
import scipy.constants as scicon

def f(x, I_S, A, B):
    return I_S - A*np.exp(B*x)

def g(x, A, B):
    return A*x**B

U_fit = np.linspace(5, 210, 20)

U1, I1 = np.genfromtxt('data/wertek1.txt', unpack=True)
U2, I2 = np.genfromtxt('data/wertek2.txt', unpack=True)
U3, I3 = np.genfromtxt('data/wertek3.txt', unpack=True)
U4, I4 = np.genfromtxt('data/wertek4.txt', unpack=True)
U5, I5 = np.genfromtxt('data/wertek5.txt', unpack=True)
U_A, I_A = np.genfromtxt('data/wertec.txt', unpack=True)

# a) Kennlinien
print("a) Kennlinien")

params, pcov = curve_fit(f, U1, I1, p0=(0.15, 1, -1))
errors = np.sqrt(np.diag(pcov))
print('Kennlinie 1')
print('I_S =', params[0], '±', errors[0])
I_S_1 = ufloat(params[0], errors[0])
print('A =', params[1], '±', errors[1])
print('B =', params[2], '±', errors[2])
plt.plot(U1, I1, 'kx', label='Messwerte')
plt.plot(U_fit, f(U_fit, *params), 'r-', label='Fit')
plt.xlim(5, 210)
plt.grid()
plt.legend()
plt.xlabel(r'$U / \text{V}$')
plt.ylabel(r'$I_\text{S} / \text{mA}$')
plt.savefig('build/plot1.pdf')
plt.clf()


params, pcov = curve_fit(f, U2, I2, p0=(0.3, 1, -1))
errors = np.sqrt(np.diag(pcov))
print('Kennlinie 2')
print('I_S =', params[0], '±', errors[0])
I_S_2 = ufloat(params[0], errors[0])
print('A =', params[1], '±', errors[1])
print('B =', params[2], '±', errors[2])
plt.plot(U2, I2, 'kx', label='Messwerte')
plt.plot(U_fit, f(U_fit, *params), 'r-', label='Fit')
plt.xlim(5, 210)
plt.grid()
plt.legend()
plt.xlabel(r'$U / \text{V}$')
plt.ylabel(r'$I_\text{S} / \text{mA}$')
plt.savefig('build/plot2.pdf')
plt.clf()


params, pcov = curve_fit(f, U3, I3, p0=(0.6, 0.6, -1))
errors = np.sqrt(np.diag(pcov))
print('Kennlinie 3')
print('I_S =', params[0], '±', errors[0])
I_S_3 = ufloat(params[0], errors[0])
print('A =', params[1], '±', errors[1])
print('B =', params[2], '±', errors[2])
plt.plot(U3, I3, 'kx', label='Messwerte')
plt.plot(U_fit, f(U_fit, *params), 'r-', label='Fit')
plt.xlim(5, 210)
plt.grid()
plt.legend()
plt.xlabel(r'$U / \text{V}$')
plt.ylabel(r'$I_\text{S} / \text{mA}$')
plt.savefig('build/plot3.pdf')
plt.clf()


params, pcov = curve_fit(f, U4, I4, p0=(1.2, 1, -1))
errors = np.sqrt(np.diag(pcov))
print('Kennlinie 4')
print('I_S =', params[0], '±', errors[0])
I_S_4 = ufloat(params[0], errors[0])
print('A =', params[1], '±', errors[1])
print('B =', params[2], '±', errors[2])
plt.plot(U4, I4, 'kx', label='Messwerte')
plt.plot(U_fit, f(U_fit, *params), 'r-', label='Fit')
plt.xlim(5, 210)
plt.grid()
plt.legend()
plt.xlabel(r'$U / \text{V}$')
plt.ylabel(r'$I_\text{S} / \text{mA}$')
plt.savefig('build/plot4.pdf')
plt.clf()

# b) Raumladung
print("b) Raumladung")

params, pcov = curve_fit(g, U5, I5, p0=(2/200**1.5, 1.5))
errors = np.sqrt(np.diag(pcov))
print('Raumladung')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])
plt.plot(U5, I5, 'kx', label='Messwerte')
plt.plot(U_fit, g(U_fit, *params), 'b-', label='Fit')
plt.xlim(5, 210)
plt.grid()
plt.legend()
plt.xlabel(r'$U / \text{V}$')
plt.ylabel(r'$I_\text{S} / \text{mA}$')
plt.savefig('build/plot5.pdf')
plt.clf()

# c) Anlaufstrom
print("c) Anlaufstrom")
I_A = I_A*100
U_k = U_A - 10**6*I_A*10**(-9)
U_A_fit = np.linspace(-0.1, 1.1, 20)
def h(x, A, B):
    return A*np.exp(B*x)

params, pcov = curve_fit(h, U_k, I_A)
errors = np.sqrt(np.diag(pcov))
print('Anlaufstrom')
print('A =', params[0], '±', errors[0])
print('B =', params[1], '±', errors[1])
plt.plot(U_k, I_A, 'kx', label='Messwerte')
plt.plot(U_A_fit, h(U_A_fit, *params), 'g-', label='Fit')
plt.xlim(-0.1, 1.1)
plt.grid()
plt.legend()
plt.xlabel(r'$U / \text{V}$')
plt.ylabel(r'$I_\text{S} / \text{nA}$')
plt.savefig('build/plot6.pdf')
plt.clf()

# Temperatur
exponent = ufloat(params[1], errors[1])
T_Kathode = - scicon.e/(scicon.k*exponent)
print("Kathodentemperatur: ", T_Kathode)

# d) Heizleistung
print("d) Heizleistung")
U = 4.3
I = 2
T1 = ((U*I -1)/(0.32*5.7*10**(-12)*0.28))**(1/4)
print("K1: T = ", T1)
U = 4.6
I = 2.1
T2 = ((U*I -1)/(0.32*5.7*10**(-12)*0.28))**(1/4)
print("K2: T = ", T2)
U = 5
I = 2.2
T3 = ((U*I -1)/(0.32*5.7*10**(-12)*0.28))**(1/4)
print("K3: T = ", T3)
U = 5.3
I = 2.3
T4 = ((U*I -1)/(0.32*5.7*10**(-12)*0.28))**(1/4)
print("K4: T = ", T4)
U = 6.2
I = 2.5
T5 = ((U*I -1)/(0.32*5.7*10**(-12)*0.28))**(1/4)
print("K5: T = ", T5)

# e) Austrittsarbeit
print("e) Austrittsarbeit")
Arbeit = -scicon.k*T1*unp.log((I_S_1*scicon.h**3)/(4*scicon.pi*0.32*10**(-4)*scicon.e*scicon.electron_mass*scicon.k**2*T1**2))
print("K1: ", Arbeit)
print(Arbeit/scicon.electron_volt)
Arbeit = -scicon.k*T2*unp.log((I_S_2*scicon.h**3)/(4*scicon.pi*0.32*10**(-4)*scicon.e*scicon.electron_mass*scicon.k**2*T2**2))
print("K2: ", Arbeit)
print(Arbeit/scicon.electron_volt)
Arbeit = -scicon.k*T3*unp.log((I_S_3*scicon.h**3)/(4*scicon.pi*0.32*10**(-4)*scicon.e*scicon.electron_mass*scicon.k**2*T3**2))
print("K3: ", Arbeit)
print(Arbeit/scicon.electron_volt)
Arbeit = -scicon.k*T4*unp.log((I_S_4*scicon.h**3)/(4*scicon.pi*0.32*10**(-4)*scicon.e*scicon.electron_mass*scicon.k**2*T4**2))
print("K4: ", Arbeit)
print(Arbeit/scicon.electron_volt)
