import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy
import scipy.constants as scicon
import math

def f(x):
   return (1/9)*(x**2-1)**2/((1-x**2)**2 + 9*x**2) #Gleichung 19

def m(x,y):
    z=0
    for a in range(0,y):
        z = z + (x[a] - np.mean(x))**2
    return 1/math.sqrt(y) * math.sqrt(1/(y-1)*z)

f_e, U_br = np.genfromtxt('data\WerteE1.txt', unpack=True) #Frequenz in Hz, Brueckenspannung in mV
f_e, U_s = np.genfromtxt('data\WerteE2.txt', unpack=True) #Frequenz in Hz, Signalspannung in mV
R2_a1, R3_a1, R4_a1 = np.genfromtxt('data\WerteA1.txt', unpack=True) #Widerstand in Ohm
R2_a2, R3_a2, R4_a2 = np.genfromtxt('data\WerteA2.txt', unpack=True) #Widerstand in Ohm
R3_b1, R4_b1, C2_b1 = np.genfromtxt('data\WerteB1.txt', unpack=True) #Widerstand in Ohm, Kapazität in nF
R3_b2, R4_b2, C2_b2 = np.genfromtxt('data\WerteB2.txt', unpack=True) #Widerstand in Ohm, Kapazität in nF
R2_b3, R3_b3, R4_b3, C2_b3 = np.genfromtxt('data\WerteB3.txt', unpack=True) #Widerstand in Ohm, Kapazität in nF
R2_c, R3_c, R4_c, L2_c = np.genfromtxt('data\WerteC.txt', unpack=True) #Widerstand in Ohm, Induktivität in mH
R2_d, R3_d, R4_d, C4_d = np.genfromtxt('data\WerteD.txt', unpack=True) #Widerstand in Ohm, Kapazität in nF

U_br = U_br * 1e-3
U_s = U_s * 1e-3
C2_b1 = C2_b1 * 1e-9
C2_b2 = C2_b2 * 1e-9
C2_b3 = C2_b3 * 1e-9
L2_c = L2_c * 1e-3
C4_d = C4_d * 1e-9

R2_a1 = unumpy.uarray(R2_a1, 0.002*R2_a1)
R2_a2 = unumpy.uarray(R2_a2, 0.002*R2_a2)
R34_a1 = R3_a1/R4_a1
R34_a2 = R3_a2/R4_a2
R34_a1 = unumpy.uarray(R34_a1, 0.005*R34_a1)
R34_a2 = unumpy.uarray(R34_a2, 0.005*R34_a2)

R43_b1 = R4_b1/R3_b1
R43_b2 = R4_b2/R3_b2
R34_b3 = R3_b3/R4_b3
R43_b3 = R4_b3/R3_b3
R43_b1 = unumpy.uarray(R43_b1, 0.005*R43_b1)
R43_b2 = unumpy.uarray(R43_b2, 0.005*R43_b2)
R34_b3 = unumpy.uarray(R34_b3, 0.005*R34_b3)
R43_b3 = unumpy.uarray(R43_b3, 0.005*R43_b3)
R2_b3 = unumpy.uarray(R2_b3, 0.03*R2_b3)
C2_b1 = unumpy.uarray(C2_b1, 0.002*C2_b1)
C2_b2 = unumpy.uarray(C2_b2, 0.002*C2_b2)
C2_b3 = unumpy.uarray(C2_b3, 0.002*C2_b3)

R2_c = unumpy.uarray(R2_c, 0.03*R2_c)
R34_c = R3_c/R4_c
R34_c = unumpy.uarray(R34_c, 0.005*R34_c)
L2_c = unumpy.uarray(L2_c, 0.002*L2_c)

R2_d = unumpy.uarray(R2_d, 0.002*R2_d)
R3_d = unumpy.uarray(R3_d, 0.03*R3_d)
R4_d = unumpy.uarray(R4_d, 0.03*R4_d)

R0_e = 332
R_e = 500
C_e = 993e-9

#a)
R13 = R2_a1 * R34_a1
R10 = R2_a2 * R34_a2

print('R10: ',np.mean(R10), '(Gauß) +/-', m(unumpy.nominal_values(R10), 3), '(Mittelwert)')

print('R13: ',np.mean(R13), '(Gauß) +/-', m(unumpy.nominal_values(R13), 3), '(Mittelwert)')


#b)
C3 = C2_b1 * R43_b1
C1 = C2_b2 * R43_b2
C8 = C2_b3 * R43_b3
R_C8 = R2_b3 * R34_b3

print('C3: ',np.mean(C3), '(Gauß) +/-', m(unumpy.nominal_values(C3), 3), '(Mittelwert)')
print('C1: ', np.mean(C1), '(Gauß) +/-', m(unumpy.nominal_values(C1), 3), '(Mittelwert)')
print('C8: ', np.mean(C8), '(Gauß) +/-', m(unumpy.nominal_values(C8), 3), '(Mittelwert)')
print('R_C8: ', np.mean(R_C8), '(Gauß) +/-', m(unumpy.nominal_values(R_C8), 3), '(Mittelwert)')

#c)
L18_c = L2_c *R34_c
R_L18_c = R2_c * R34_c

print('L18: ', L18_c)
print('R_L18: ', R_L18_c)

#d)
L18_d = R2_d * R3_d * C4_d
R_L18_d = R2_d * R3_d / R4_d

print('L18_d : ', np.mean(L18_d), '(Gauß) +/-', m(unumpy.nominal_values(L18_d), 3), '(Mittelwert)')
print('R_L18_d: ', np.mean(R_L18_d), '(Gauß) +/-', m(unumpy.nominal_values(R_L18_d), 3), '(Mittelwert)')

#e)
w0_e = 1/(R0_e * C_e *2 * math.pi)
O_e = f_e / w0_e
U_br_eff = U_br/(2*2**(0.5))
plt.plot(O_e, U_br_eff/U_s, 'kx', label='Messwerte')
plt.plot(np.linspace(0,70,10000), f(np.linspace(0,70,10000)), 'r-', label='Theorie')
plt.grid()
plt.legend()
plt.xlabel(r'$\nu / \nu_0$')
plt.ylabel(r'$\frac{U_{Br,eff}}{U_s}$', rotation=0)
plt.xlim(0.03, 69)
plt.ylim(0,0.13)
plt.xscale('log')
plt.savefig('build\plot1.pdf')
print('w_0: ', 1/(R0_e * C_e))
print('v_0: ', w0_e)

#f)
U_br_min = U_br[17] #14mV
U2 = U_br_min / f(2)
U1 = U_s[17] #3.88V
k = U2 / U1
print('U2: ', U2)
print('k: ',k)