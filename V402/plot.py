import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math

def f(x, a, b):     #Erste n Funktion
    return a + (b / x**2)

def g(x, c, d):     #Zweite n Funktion
    return c - (d * x**2)

def h(x, m ,n):
    return m*x+n

t_1, t_2 = np.genfromtxt("data/theta.txt", unpack=True) #Thetawinkel
p_1, p_2 = np.genfromtxt("data/phi.txt", unpack=True) #Phiwinkel

phi = (p_2 - p_1) / 2
theta = 180 - (t_1 - t_2)

phi_m = np.mean(phi)

n = np.sin(np.deg2rad((theta + phi_m) / 2)) / np.sin(np.deg2rad(phi_m / 2))

phi_err = ufloat(phi_m,np.std(phi))


s = [643.85, 578.02, 546.07,491.61, 467.03, 435.83, 407.78] #Wellenlängen von rot nach violett

print("theta: ", theta)
print("phi: ", phi_err)
print("n: ", n,"\n")

x = np.linspace(400,620,10000)
params1, pcov1 = curve_fit(f,s,n**2)
params2, pcov2 = curve_fit(g,s,n**2)
print("Fit fuer 1/lambda^2 \n")
print("A0: ", params1[0] ,"+-", pcov1[0][0])
print("A: ", params1[1],"+-", pcov1[1][0],"\n")
print("Fit fuer lambda^2 \n")
print("A0: ", params2[0] ,"+-", pcov2[0][0])
print("A: ", params2[1],"+-", pcov2[1][0],"\n")

plt.plot(x, f(x, *params1), 'r', label="Fit mit f(x)") #passt viel besser
plt.plot(x, g(x, *params2), 'g', label="Fit mit g(x)")
plt.plot(s, n**2, 'bx', label="Messwerte")
plt.grid()
plt.legend()
plt.xlabel("Wellenläng in nm")
plt.ylabel("n²")
plt.savefig("build/plot1.pdf")
plt.clf()

i = [0,1,2,3,4,5,6]
abw = 0
abw2 = 0
for x in i:
    abw += (n[x]**2 - params1[0] - params1[1]/(s[x]**2))**2

for x in i:
    abw2 += (n[x]**2 - params2[0] - params2[1]*(s[x]**2))**2

abw_q = 1/(7-2) * abw
abw_q2 = 1/(7-2) * abw2

print("Abweichungsquadrat (guter Fit): ", abw_q)
print("Abweichungsquadrat (schlechter Fit): ", abw_q2)

s_q = s
for x in i:
    s_q[x] = s[x]**2

A = ufloat(params1[0],pcov1[0][0])
A2 = ufloat(params1[1],-pcov1[1][0])


def n_fit(x):
    return (A + A2/(x**2))**(0.5)


#ABELSCHE ZAHL
n_C = n_fit(656)
n_D = n_fit(589)
n_F = n_fit(486)

nu = (n_D - 1)/(n_F - n_C)
print("n_C: ", n_C)
print("n_D: ", n_D)
print("n_F: ", n_F)
print("Abelsche Zahl: ", nu)

#Auflösungsvermögen
def abl_n_fit(x):
    return A2/(x**3*(A + A2/(x**2))**(0.5))

print("Aufloesung C: ", 3e7*abl_n_fit(656)) #3e7 sind 3cm in nm
print("Aufloesung D: ", 3e7*abl_n_fit(589))
print("Aufloesung F: ", 3e7*abl_n_fit(486))

#Absorbtionsstelle

absorp= (A2 / (A -1))**(0.5)
print("Absorptionsstelle: ", absorp)