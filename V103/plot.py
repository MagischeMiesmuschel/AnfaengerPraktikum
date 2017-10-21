import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

xe, yre, yqe = np.genfromtxt('werte1.txt', unpack=True)
xz, yqz = np.genfromtxt('werte2.txt', unpack=True)

def f(x, a, b):
    return a*x + b

Elit = 18e10 # Elastizitaetsmodul Literaturwert

#zylindrischer Stab einseitige Einspannung
L = 0.5 # in m, Stablänge
M = 0.2488 # in kg, angehängte Masse
B = 7*1e-3 # in m, Breite des Stabes
xe *= 1e-3 # von mm auf m
yre *= 1e-3 # von mm auf m

xe = L*xe**2-xe**3/3 # D(x)/y soll gegen diese Funktion aufgetragen werden

params, pcov = curve_fit(f,xe,yre)
errors = np.sqrt(np.diag(pcov))

I = (np.pi/4)*(B/2)**4 # Trägheitsmoment
E = (M*9.81)/(2*I*params[0]) # Elastizitaetsmodul

print('Parameter fuer zylindrischen Stab bei einseitiger Einspannung:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(1-E/Elit)

plt.plot(xe*1e3,yre*1e3, 'kx', label='Messwerte')
plt.plot(xe*1e3, f(xe, *params)*1e3, 'r-', label='Fit')
plt.xlabel(r'$(Lx^2-\frac{x^3}{3}) \:/\: 10^{3} \: m^3}$')
plt.ylabel(r'$D(x) \:/\: 10^{3} \: m$')
plt.legend(loc='best')

plt.savefig('rundeinseitg.pdf')
plt.clf()

#quadratischer Stab einseitige Einspannung
L = 0.545 # in m, Stablänge
M = 0.2488 # in kg, angehängte Masse
B = 8*1e-3 # in m, Breite des Stabes
yqe *= 1e-3 # von mm auf m

params, pcov = curve_fit(f,xe,yqe)
errors = np.sqrt(np.diag(pcov))

I = (B**4)/12 # Trägheitsmoment
E = (M*9.81)/(2*I*params[0]) # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei einseitiger Einspannung:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(1-E/Elit)

plt.plot(xe*1e3,yqe*1e3, 'kx', label='Messwerte')
plt.plot(xe*1e3, f(xe, *params)*1e3, 'r-', label='Fit')
plt.xlabel(r'$(Lx^2-\frac{x^3}{3}) \:/\: 10^{3} \: m^3}$')
plt.ylabel(r'$D(x) \:/\: 10^{3} \: m$')
plt.legend(loc='best')

plt.savefig('quadeinseitg.pdf')
plt.clf()

#quadratischer Stab zweiseitig Einspannung
L = 0.56 # in m, Stablänge
M = 3.550 # in kg, angehängte Masse
B = 8*1e-3 # in m, Breite des Stabes
xz *= 1e-3 # von mm auf m
yqz *= 1e-3 # von mm auf m

xz = 3*(L**2)*xz-4*xz**3 # D(x)/y soll gegen diese Funktion aufgetragen werden

params, pcov = curve_fit(f,xz,yqz)
errors = np.sqrt(np.diag(pcov))

I = (B**4)/12 # Trägheitsmoment
E = (M*9.81)/(48*I*params[0]) # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei zweiseitiger Einspannung:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(1-E/Elit)

plt.plot(xz*1e3,yqz*1e3, 'kx', label='Messwerte')
plt.plot(xz*1e3, f(xz, *params)*1e3, 'r-', label='Fit')
plt.xlabel(r'$(3L^2x-4x^3) \:/\: 10^{3} \: m^3}$')
plt.ylabel(r'$D(x) \:/\: 10^{3} \: m$')
plt.legend(loc='best')

plt.savefig('quadzweiseitg.pdf')
plt.clf()
