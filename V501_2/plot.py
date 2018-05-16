import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math

def f(x, m, n):
    return m*x+n

U_d1, U_d2, U_d3, U_d4, U_d5, D = np.genfromtxt('data/werte1_a.txt', unpack=True)
U_Ba1 = ([200, 230, 250, 280, 300])
U_Ba1_inv=([1/200, 1/230, 1/250, 1/280, 1/300]) 
nu, n = np.genfromtxt('data/werte1_b.txt', unpack=True) #Herz
nu_neu = nu*n
y_ausl = 0.013  #Meter
p = 0.019
d = 0.0038
L = 0.143
N = 20
R = 0.282

I1, I2 = np.genfromtxt('data/werte2_a.txt', unpack=True) #Amper

print("U_d1: ", U_d1)
print("U_d2: ", U_d2)
print("U_d3: ", U_d3)
print("U_d4: ", U_d4)
print("U_d5: ", U_d5)
print("d: ", d)
print("nu: ", nu)
print("I1: ", I1)
print("I2: ", I2)

x = np.linspace(-20, 35, 1000)

plt.plot(U_d1, D, 'gx', label="U = 200V")
plt.plot(U_d2, D, 'rx', label="U = 230V")
plt.plot(U_d3, D, 'bx', label="U = 250V")
plt.plot(U_d4, D, 'yx', label="U = 280V")
plt.plot(U_d5, D, 'kx', label="U = 300V")
params1, pcov1 = curve_fit(f, U_d1, D)
params2, pcov2 = curve_fit(f, U_d2, D)
params3, pcov3 = curve_fit(f, U_d3, D)
params4, pcov4 = curve_fit(f, U_d4, D)
params5, pcov5 = curve_fit(f, U_d5, D)
print("m1: ", params1[0], params1[1])
print("m2: ", params2[0], params2[1])
print("m3: ", params3[0], params3[1])
print("m4: ", params4[0], params4[1])
print("m5: ", params5[0], params5[1])
plt.plot(x,f(x, *params1),'g', label="Fit1")
plt.plot(x,f(x, *params2),'r', label="Fit2")
plt.plot(x,f(x, *params3),'b', label="Fit3")
plt.plot(x,f(x, *params4),'y', label="Fit4")
plt.plot(x,f(x, *params5),'k', label="Fit5")
plt.grid()
plt.legend()
plt.xlabel('U in V')
plt.ylabel('D in m')
plt.axis([-15,35,-0.01,0.05])
plt.savefig('build/plot1.pdf')
plt.clf()

m = ([params1[0], params2[0], params3[0], params4[0], params5[0]])

x = np.linspace(0.003, 0.006, 1000)

plt.plot(U_Ba1_inv, m, 'kx', label="Messwerte")
params, pcov = curve_fit(f, U_Ba1_inv, m)
print("a: ", params[0])
print("pL/2d: ", (p*L)/(2*d))
plt.plot(x,f(x, *params), label="Fit")
plt.legend()
plt.grid()
plt.axis([0.0032, 0.0052, 0.001, 0.0018])
plt.xlabel("1/$U_B$ in 1/V")
plt.ylabel("D/$U_d$ in m/V")
plt.savefig('build/plot2.pdf')
plt.clf()

a_abw = (1-(params[0]/(p*L/(2*d))))*100
print("a_abw: ", a_abw, "%")

nu_mittel = (nu[0]*n[0] + nu[1]*n[1] + nu[2]*n[2] + nu[3]*n[3]) / 4

print("nu_mittel: ", nu_mittel)
print("nu_neu: ", nu_neu)

nu_abw = ((nu_mittel - nu_neu[0]) + (nu_neu[1] - nu_mittel) + (nu_neu[2] - nu_mittel) + (nu_mittel - nu_neu[3]))/3

print("nu_abw: ", nu_abw)

mu_null = 4*math.pi*(10**(-7))

B1 = mu_null * 8 / (math.sqrt(125)) * N * I1 / R * (10**3) #mT
B2 = mu_null * 8 / (math.sqrt(125)) * N * I2 / R * (10**3)

x = np.linspace(-1, 3, 1000)
D_neu = D / ( L * L + D * D)

plt.plot(B1, D_neu, 'gx')
plt.plot(B2, D_neu, 'kx')
paramsB1, pcovB1 = curve_fit(f, B1, D_neu)
paramsB2, pcovB2 = curve_fit(f, B2, D_neu)
plt.plot(x,f(x, *paramsB1), 'g')
plt.plot(x,f(x, *paramsB2), 'k')
plt.axis([-0.05,0.3,-0.05,3])
plt.grid()
plt.legend()
plt.xlabel("B in mT")
plt.ylabel("D/(L²+D²)")
plt.savefig('build/plot3.pdf')
print("paramsB1: ", paramsB1)
print("paramsB2: ", paramsB2)
plt.clf()

print(B1)
print(B2)

spezLadung1 = paramsB1[0]*(10**3)*paramsB1[0]*(10**3)*8*250
spezLadung2 = paramsB2[0]*(10**3)*paramsB2[0]*(10**3)*8*350
print("spezLadung: ", spezLadung1, "   spezLadung2: ", spezLadung2)
spezMittel = (spezLadung1 + spezLadung2)/2
spezAbw = (spezMittel-spezLadung1 + spezLadung2-spezMittel)

print("spezAbw: ", spezAbw)
print("spezMittel: ", spezMittel)