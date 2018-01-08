import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
from scipy.optimize import curve_fit
import scipy.constants as scicon

t, T1, T2, pa, pb, P = np.genfromtxt('data/werte.txt', unpack=True)

def f(t, A, B, C):
    return A*t**2 + B*t + C

# Zeit, Druck und Temperatur anpassen

t = t*60
pa = pa + 1
pb = pb + 1
pa = pa*1e5
pb = pb*1e5
T1 = T1 + 273.15
T2 = T2 + 273.15

# Temperatur Fitten

params1, pcov1 = curve_fit(f, t, T1)
errors1 = np.sqrt(np.diag(pcov1))
print("Approximation der Temperaturverlaeufe")
print("Parameter fuer T1")
print('A = ', params1[0], '+/-', errors1[0])
print('B = ', params1[1], '+/-', errors1[1])
print('C = ', params1[2], '+/-', errors1[2])

params2, pcov2 = curve_fit(f, t, T2)
errors2 = np.sqrt(np.diag(pcov2))
print("Parameter fuer T2")
print('A = ', params2[0], '+/-', errors2[0])
print('B = ', params2[1], '+/-', errors2[1])
print('C = ', params2[2], '+/-', errors2[2])
print()

# Temperatur Plotten

t_fit = np.linspace(60, 1100, 50)
plt.plot(t, T1, 'rx', label='Messwerte für T1')
plt.plot(t, T2, 'bx', label='Messwerte für T2')
plt.plot(t_fit, f(t_fit, *params1), 'r-', label='Fit für T1')
plt.plot(t_fit, f(t_fit, *params2), 'b-', label='Fit für T2')
plt.xlim(0, 1100)
plt.xlabel(r'$t$ / s')
plt.ylabel(r'$T$ / K')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/plot1.pdf')
plt.clf()

# Differentialquotienten

# Ableitungsfunktion: dT/dt = 2*A*t + B
count = [1, 2, 3, 4]

print("Differentialquotient fuer T1")
A1 = ufloat(params1[0], errors1[0])
B1 = ufloat(params1[1], errors1[1])
for x in count:
    dT1dt = 2*A1*x*180 + B1
    print("t = ", x*180, "dT1/dt: ", dT1dt.n, dT1dt.s)

print("Differentialquotient fuer T2")
A2 = ufloat(params2[0], errors2[0])
B2 = ufloat(params2[1], errors2[1])
for x in count:
    dT2dt = 2*A2*x*180 + B2
    print("t = ", x*180, "dT2/dt: ", dT2dt.n, dT2dt.s)
print()

# Güteziffer

c_w = 4.182*1e3 # J/kgK
m_w = 3 #kg
c_k_m_k = 660 # J/K

print("Guete")
for x in count:
    guete = (c_w*m_w + c_k_m_k)*(2*A1*x*180 + B1)/P[x*3-1]
    print("t = ", x*180, "Guete: ", guete)
print()

# Verdampfungswaerme

def g(x, m, n):
    return m*x + n
p_0 = 1e5 # Umgebungsdruck

# für T1 und pb

params3, pcov3 = curve_fit(g, 1/T1, np.log(pb/p_0) )
errors3 = np.sqrt(np.diag(pcov3))
m = ufloat(params3[0], errors3[0])
L = -m*scicon.R
print("Verdampfungswaerme fuer T1")
print('m = ', params3[0], '+/-', errors3[0])
print('n = ', params3[1], '+/-', errors3[1])
print("Verdampfungswaerme pro Mol: ", L)
molaremasse_cl2f2c = 0.12091 # kg/mol
print("molaremasse_cl2f2c: ", molaremasse_cl2f2c, "kg/mol")
print("Verdampfungswaerme pro kg: ", L/molaremasse_cl2f2c)
print()

T1_fit = np.linspace(3.05*1e-3,3.45*1e-3,20)
plt.plot(1/T1*1e3, np.log(pb/p_0), 'kx', label='Messwerte')
plt.plot(T1_fit*1e3, g(T1_fit, *params3), 'r-', label='Fit')
plt.xlabel(r'$1/T_1$ / ($10^{-3}\cdot$1/s)')
plt.ylabel(r'$ln(p_b/p_0)$')
plt.xlim(3.05, 3.45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/plot2.pdf')
plt.clf()

#Massendurchsatz

print("Massendurchsatz")
for x in count:
    m_durchsatz = (c_w*m_w + c_k_m_k)*(2*A2*x*180 + B2)/L
    print("t = ", x*180, "Massendurchsatz in Mol: ", m_durchsatz)
    print("Massendurchsatz:", m_durchsatz*molaremasse_cl2f2c)
    # mechansiche Leistung
    m_durchsatz = (c_w*m_w + c_k_m_k)*(2*A2*x*180 + B2)/L
    dichte = (273.15*pa[x*3-1]*5.51)/(T2[x*3-1]*1e5)
    N_mech = 1/(1.14 - 1)*(pb[x*3-1]*((pa[x*3-1]/pb[x*3-1])**(1/1.14))-pa[x*3-1])*(1/dichte)*m_durchsatz*molaremasse_cl2f2c
    print("mechansiche Leistung")
    print(N_mech)
    print("Wirkunsgrad")
    print(N_mech/P[x*3-1])
    print()
print()
