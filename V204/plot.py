import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import uncertainties.unumpy as unp

def min(t, x = np.array([])):
    y = np.array([])
    z = np.array([])
    count = 2
    while count <= t.size-1:
        if x[count-2] >= x[count-1] and x[count] > x[count-1]:
            y = np.append(y, x[count-1])
            z = np.append(z, count -1)
        count += 1
    return np.array([z, y])

def max(t, x = np.array([])):
    y = np.array([])
    z = np.array([])
    count = 2
    while count <= t.size-1:
        if x[count-2] < x[count-1] and x[count] <= x[count-1]:
            y = np.append(y, x[count-1])
            z = np.append(z, count -1)
        count += 1
    return np.array([z, y])

# statische Methode //////////////////////////////////////////////////////////////////////
print("statische Methode")
print()

t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('data/GLXportRun1.txt', unpack=True)
t = t*5 # Umrechnung in Sekunden, Abtastrate

plt.plot(t, T1, "b.", ms=3, label="Messwerte T1")
plt.plot(t, T4, "g.", ms=3, label="Messwerte T4")
plt.xlabel(r'Zeit in $s$')
plt.ylabel(r'Temperatur in °C')
plt.grid()
plt.legend()
plt.savefig('build/Plot1.pdf')
plt.clf()

plt.plot(t, T5, "b.", ms=3, label="Messwerte T5")
plt.plot(t, T8, "g.", ms=3, label="Messwerte T8")
plt.xlabel(r'Zeit in $s$')
plt.ylabel(r'Temperatur in °C')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/Plot2.pdf')
plt.clf()

print("Temperatur bei 700 s", 28.20, 27.54, 29.26, 25.32)
print()

plt.plot(t, T7 - T8, "b.", ms=3, label="Messwerte T7 - T8")
plt.plot(t, T2 - T1, "g.", ms=3, label="Messwerte T2 - T1")
plt.xlabel(r'Zeit in $s$')
plt.ylabel(r'Temperatur in °C')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('build/Plot6.pdf')
plt.clf()

#dynamische Methode 1 //////////////////////////////////////////////////////////////////////
print("dynamische Methode 1")
print()

t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('data/GLXportRun2.txt', unpack=True)
t = t*2 # Umrechnung in Sekunden, Abtastrate

print("Messing, breit")
print()

# Maxima und Minima
T1_min = min(t, T1)
T2_min = min(t, T2)
T1_max = max(t, T1)
T2_max = max(t, T2)

# Amplitude
count = 1
A1 = np.array([])
A2 = np.array([])
while count <= T1_max[1,].size-1: # size-1 weil für letzten Tiefpunkt kein Tiefpunkt zum Vergleich vorhanden ist
    A1 = np.append(A1, (T1_max[1,count-1] - T1_min[1,count-1])/2 - (T1_min[1,count] - T1_min[1,count-1])/2) #von Tiefpunkt zu Hochpunktdurch 2 minus halbe Steigung von Tiefpunkt zu Tiefpunkt
    A2 = np.append(A2, (T2_max[1,count-1] - T2_min[1,count-1])/2 - (T2_min[1,count] - T2_min[1,count-1])/2)
    count += 1
print("Amplitude")
print("A1: ", A1)
print("A2: ", A2)

#Phasendifferenz
count = 1
deltaT1T2 = np.array([])
while count <= T1_max[0,].size:
    deltaT1T2 = np.append(deltaT1T2, T1_min[0,count-1]*2 - T2_min[0,count-1]*2) # Phasendifferenz zwischen zwei Tiefpunkten
    count += 1
print("Phasendifferenz: ", deltaT1T2)

# Wärmeleitfähigkeit
A1 = ufloat(np.mean(A1), np.std(A1))
A2 = ufloat(np.mean(A2), np.std(A2))
deltaT1T2 = ufloat(np.mean(deltaT1T2), np.std(deltaT1T2))

kappaT1T2 = (8520*385*0.03**2)/(2*deltaT1T2*unp.log(A2/A1))
print("Wärmeleitfähigkeit: ", kappaT1T2)
print()

# Messing breit
plt.plot(t, T1, "b.", ms=3, label="Messwerte T1")

plt.plot(T1_min[0,]*2,T1_min[1,], "kx")
plt.plot(T2_min[0,]*2,T2_min[1,], "kx")
plt.plot(T1_max[0,]*2,T1_max[1,], "rx")
plt.plot(T2_max[0,]*2,T2_max[1,], "kx")

plt.plot(t, T2, "g.", ms=3, label="Messwerte T2")
plt.xlabel(r'Zeit in $s$')
plt.ylabel(r'Temperatur in °C')
plt.grid()
plt.legend()
plt.savefig('build/Plot3.pdf')
plt.clf()

print("Aluminium")
print()

# Maxima und Minima
T5_min = min(t, T5)
T6_min = min(t, T6)
T5_max = max(t, T5)
T6_max = max(t, T6)

# Amplitude
count = 1
A5 = np.array([])
A6 = np.array([])
while count <= T5_max[1,].size-1: # size-1 weil für letzten Tiefpunkt kein Tiefpunkt zum Vergleich vorhanden ist
    A5 = np.append(A5, (T5_max[1,count-1] - T5_min[1,count-1])/2 - (T5_min[1,count] - T5_min[1,count-1])/2) #von Tiefpunkt zu Hochpunktdurch 2 minus halbe Steigung von Tiefpunkt zu Tiefpunkt
    A6 = np.append(A6, (T6_max[1,count-1] - T6_min[1,count-1])/2 - (T6_min[1,count] - T6_min[1,count-1])/2)
    count += 1
print("Amplitude")
print("A5: ", A5)
print("A6: ", A6)

#Phasendifferenz
count = 1
deltaT5T6 = np.array([])
while count <= T5_max[0,].size:
    deltaT5T6 = np.append(deltaT5T6, T5_min[0,count-1]*2 - T6_min[0,count-1]*2) # Phasendifferenz zwischen zwei Tiefpunkten
    count += 1
print("Phasendifferenz: ", deltaT5T6)

# Wärmeleitfähigkeit
A5 = ufloat(np.mean(A5), np.std(A5))
A6 = ufloat(np.mean(A6), np.std(A6))
deltaT5T6 = ufloat(np.mean(deltaT5T6), np.std(deltaT5T6))

kappaT5T6 = (2800*830*0.03**2)/(2*deltaT5T6*unp.log(A6/A5))
print("Wärmeleitfähigkeit: ", kappaT5T6)
print()

# Aluminium
plt.plot(t, T5, "b.", ms=3, label="Messwerte T5")
plt.plot(t, T6, "g.", ms=3, label="Messwerte T6")
plt.xlabel(r'Zeit in $s$')
plt.ylabel(r'Temperatur in °C')
plt.grid()
plt.legend()
plt.savefig('build/Plot4.pdf')
plt.clf()

#dynamische Methode 2 //////////////////////////////////////////////////////////////////////
print("dynamische Methode 2")
print()

t, T1, T2, T3, T4, T5, T6, T7, T8 = np.genfromtxt('data/GLXportRun3.txt', unpack=True)
t = t*2 # Umrechnung in Sekunden, Abtastrate

print("Edelstahl")
print()

# Maxima und Minima
T7_min = min(t, T7)
T8_min = np.array([[18, 122, 222, 322, 423, 522, 629, 727, 826, 926],[28.77, 30.17, 31.03, 31.76, 32.43, 32.89, 33.05, 33.38, 33.89, 34.35]])
T7_max = max(t, T7)
T8_max = np.array([[93, 186, 281, 381, 482, 578, 684, 781, 880, 983],[30.33, 31.3, 32.06, 32.71, 33.3, 33.59, 33.78, 34.24, 34.74, 35.16]])

# Amplitude
count = 1
A7 = np.array([])
A8 = np.array([])
while count <= T7_max[1,].size-1: # size-1 weil für letzten Tiefpunkt kein Tiefpunkt zum Vergleich vorhanden ist
    A7 = np.append(A7, (T7_max[1,count-1] - T7_min[1,count-1])/2 - (T7_min[1,count] - T7_min[1,count-1])/2) #von Tiefpunkt zu Hochpunktdurch 2 minus halbe Steigung von Tiefpunkt zu Tiefpunkt
    A8 = np.append(A8, (T8_max[1,count-1] - T8_min[1,count-1])/2 - (T8_min[1,count] - T8_min[1,count-1])/2)
    count += 1
print("Amplitude")
print("A7: ", A7)
print("A8: ", A8)

#Phasendifferenz
count = 1
deltaT7T8 = np.array([])
while count <= T7_max[0,].size:
    deltaT7T8 = np.append(deltaT7T8, T7_min[0,count-1]*2 - T8_min[0,count-1]*2) # Phasendifferenz zwischen zwei Tiefpunkten
    count += 1
print("Phasendifferenz: ", deltaT7T8)

# Wärmeleitfähigkeit
A7 = ufloat(np.mean(A7), np.std(A7))
A8 = ufloat(np.mean(A8), np.std(A8))
deltaT7T8 = ufloat(np.mean(deltaT7T8), np.std(deltaT7T8))

kappaT7T8 = (8000*400*0.03**2)/(2*deltaT7T8*unp.log(A7/A8))
print("Wärmeleitfähigkeit: ", kappaT7T8)
print()

# Edelstahl
plt.plot(t, T7, "b.", ms=3, label="Messwerte T7")
plt.plot(t, T8, "g.", ms=3, label="Messwerte T8")
plt.xlabel(r'Zeit in $s$')
plt.ylabel(r'Temperatur in °C')
plt.grid()
plt.legend()
plt.savefig('build/Plot5.pdf')
plt.clf()
