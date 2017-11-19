import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

xe, yre, yqe = np.genfromtxt('werte1.txt', unpack=True)
xzl, yqzl = np.genfromtxt('werte2.txt', unpack=True)
xzr, yqzr = np.genfromtxt('werte3.txt', unpack=True)

def f(x, a, b):
    return a*x + b

Elit = 7e10 # Elastizitaetsmodul Literaturwert

#zylindrischer Stab einseitige Einspannung
L = 0.5 # in m, Stablänge
M = 0.2488 # in kg, angehängte Masse
B = 7*1e-3 # in m, Breite des Stabes
xe *= 1e-3 # von mm auf m
yre *= 1e-3 # von mm auf m

xe = L*xe**2-xe**3/3 # D(x)/y soll gegen diese Funktion aufgetragen werden

params, pcov = curve_fit(f,xe,yre)
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (np.pi/4)*(B/2)**4 # Trägheitsmoment
E = (M*9.81)/(2*I*a) # Elastizitaetsmodul

print('Parameter fuer zylindrischen Stab bei einseitiger Einspannung:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(E/Elit)

plt.plot(xe*1e3,yre*1e3, 'kx', label='Messwerte')
plt.plot(xe*1e3, f(xe, *params)*1e3, 'r-', label='Fit')
plt.xlabel(r'$(Lx^2-\frac{x^3}{3}) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
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

a = ufloat(params[0], errors[0])

I = (B**4)/12 # Trägheitsmoment
E = (M*9.81)/(2*I*a) # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei einseitiger Einspannung:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(E/Elit)

plt.plot(xe*1e3,yqe*1e3, 'kx', label='Messwerte')
plt.plot(xe*1e3, f(xe, *params)*1e3, 'r-', label='Fit')
plt.xlabel(r'$(Lx^2-\frac{x^3}{3}) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('quadeinseitg.pdf')
plt.clf()

#quadratischer Stab zweiseitig Einspannung links
L = 0.56 # in m, Stablänge
M = 3.550 # in kg, angehängte Masse
B = 8*1e-3 # in m, Breite des Stabes
xzl *= 1e-3 # von mm auf m
yqzl *= 1e-3 # von mm auf m

xzl = 3*(L**2)*xzl-4*xzl**3 # D(x)/y soll gegen diese Funktion aufgetragen werden

params, pcov = curve_fit(f,xzl,yqzl)
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (B**4)/12 # Trägheitsmoment
E = (M*9.81)/(48*I*a) # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei zweiseitiger Einspannung links:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(E/Elit)

plt.plot(xzl*1e3,yqzl*1e3, 'kx', label='Messwerte')
plt.plot(xzl*1e3, f(xzl, *params)*1e3, 'r-', label='Fit')
plt.xlabel(r'$(3L^2x-4x^3) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('quadzweiseitg_links.pdf')
plt.clf()

#quadratischer Stab zweiseitig Einspannung rechts
L = 0.56 # in m, Stablänge
M = 3.550 # in kg, angehängte Masse
B = 8*1e-3 # in m, Breite des Stabes
xzr *= 1e-3 # von mm auf m
yqzr *= 1e-3 # von mm auf m

xzr = 4*xzr**3-12*L*xzr**2+9*(L**2)*xzr-L**3  # D(x)/y soll gegen diese Funktion aufgetragen werden

params, pcov = curve_fit(f,xzr,yqzr)
errors = np.sqrt(np.diag(pcov))

a = ufloat(params[0], errors[0])

I = (B**4)/12 # Trägheitsmoment
E = (M*9.81)/(48*I*a) # Elastizitaetsmodul

print('Parameter fuer quadratischen Stab bei zweiseitiger Einspannung rechts:')
print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print('Elastizitaetsmodul:')
print(E)
print('Abweichung:')
print(E/Elit)

plt.plot(xzr*1e3,yqzr*1e3, 'kx', label='Messwerte')
plt.plot(xzr*1e3, f(xzr, *params)*1e3, 'r-', label='Fit')
plt.xlabel(r'$(4x^3-12Lx^2+9L^2x-L^3) \:/\: (10^{-3} \: m^{3})$')
plt.ylabel(r'$D(x) \:/\: (10^{-3} \: m)$')
plt.legend(loc='best')

plt.savefig('quadzweiseitg_rechts.pdf')
plt.clf()
