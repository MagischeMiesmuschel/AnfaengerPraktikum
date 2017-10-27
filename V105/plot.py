import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon

def f(x, a, b):
    return a*x + b

def B(I):
    return 195*scicon.mu_0*(I*0.109**2)/(0.109**2+(0.138/2)**2)**(3/2)

r_kugel = 0.0255 # in m, Radius Kugel
l_stift = 0.012 # in m, Länge Stift
m_kugel = 0.1422 # in kg Masse Kugel
J_kugel = 2/5*m_kugel*r_kugel**2 # Trägheitsmoment der Kugel
m_hebel = 0.0014 # in kg, Masse der Kugel


r_hebel, Ig = np.genfromtxt('werte1.txt', unpack=True)
Io, To = np.genfromtxt('werte2.txt', unpack=True)
Ik, Tk = np.genfromtxt('werte3.txt', unpack=True)

# Umwandeln in SI-Einheiten

r_hebel = r_hebel*1e-2

# Periodendauer

To = To/10
Tk = Tk/10

# Messfehler
T_error = np.ones(10)
T_error = 0.01*T_error
r_error = np.ones(10)
r_error = 1e-3*r_error
I_error = np.ones(10)
I_error = 0.1*I_error

print('Messfehler:')
print(r_error)
print(I_error)
print(T_error)

# Berechnung mit Gravitation
# Variablen berechnen, die gegeneinander aufgetragen werden sollen
r = r_hebel + r_kugel + l_stift

count = range(10)
Bg = np.zeros(10)
Bg_error = np.zeros(10)

for x in count:
    Igu = ufloat(Ig[x], I_error[x])
    Bgu = B(Igu)
    Bg[x]= Bgu.n
    Bg_error[x] = Bgu.s

print('Bg: ', Bg)
print('Bg_error: ', Bg_error)

# Ausgleichsgerade und PLots

params,pcov = curve_fit(f, r, Bg, sigma = Bg_error)
errors = np.sqrt(np.diag(pcov))
print('a: ', params[0], errors[0], sep='\n')
print('b: ', params[1], errors[1], sep='\n')

a = ufloat(params[0],errors[0])

mu_dipol = m_hebel*scicon.g/a
print('mu_dipol: ', mu_dipol)

plt.errorbar(r, Bg, xerr=r_error, yerr=Bg_error, fmt='k.', label='Messdaten')
plt.plot(r, f(r, *params), 'b-', label='Fit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot1.pdf')
plt.clf()

# Bestimmung über Schwingungsdauer
# Variablen berechnen, die gegeneinander aufgetragen werden sollen

count = range(10)
Bo = np.zeros(10)
Bo_error = np.zeros(10)
T_square = np.zeros(10)
T_square_error = np.zeros(10)

for x in count:
    Iou = ufloat(Io[x], I_error[x])
    Bou = 1/B(Iou)
    Bo[x]= Bou.n
    Bo_error[x] = Bou.s

    Tou = ufloat(To[x], T_error[x])
    Tou_square = Tou**2
    T_square[x] = Tou_square.n
    T_square_error[x] = Tou_square.s

print('Bo: ', 1/Bo)
print('Bo_error: ', 1/Bo_error)
print('T_square: ', T_square)
print('T_square_error: ', T_square_error)

# Ausgleichsgerade und PLots

params,pcov = curve_fit(f, Bo, T_square, sigma = T_square_error)
errors = np.sqrt(np.diag(pcov))
print('a: ', params[0], errors[0], sep='\n')
print('b: ', params[1], errors[1], sep='\n')

a = ufloat(params[0],errors[0])

mu_dipol = 4*np.pi**2*J_kugel/a
print('mu_dipol: ', mu_dipol)

plt.errorbar(Bo, T_square, xerr=Bo_error, yerr=T_square_error, fmt='k.', label='Messdaten')
plt.plot(Bo, f(Bo, *params), 'b-', label='Fit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot2.pdf')
plt.clf()

# Bestimmung über Präzession
# Variablen berechnen, die gegeneinander aufgetragen werden sollen

count = range(10)
Bk = np.zeros(10)
Bk_error = np.zeros(10)
T_reci = np.zeros(10)
T_reci_error = np.zeros(10)

for x in count:
    Iku = ufloat(Ik[x], I_error[x])
    Bku = B(Iku)
    Bk[x]= Bku.n
    Bk_error[x] = Bku.s

    Tku = ufloat(Tk[x], T_error[x])
    Tku_reci = 1/Tku
    T_reci[x] = Tku_reci.n
    T_reci_error[x] = Tku_reci.s

print('Bk: ', Bk)
print('Bk_error: ', Bk_error)
print('T_reci: ', T_reci)
print('T_reci_error: ', T_reci_error)

# Ausgleichsgerade und PLots

params,pcov = curve_fit(f, Bk, T_reci)
errors = np.sqrt(np.diag(pcov))
print('a: ', params[0], errors[0], sep='\n')
print('b: ', params[1], errors[1], sep='\n')

a = ufloat(params[0],errors[0])

mu_dipol = a*2*np.pi*J_kugel*5.5
print('mu_dipol: ', mu_dipol)

plt.errorbar(Bk, T_reci, xerr=Bk_error, yerr=T_reci_error, fmt='k.', label='Messdaten')
plt.plot(Bk, f(Bk, *params), 'b-', label='Fit')
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot3.pdf')
plt.clf()
