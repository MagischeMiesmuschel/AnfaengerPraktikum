import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math 

def f(x, m, n):
    return -m*x+n

def g(x, m, n):
    return n * math.exp(-m * x)

N_null = 178
t_null = 900
N_u = N_null/t_null
t_1, N_1 = np.genfromtxt("data/WerteA.txt", unpack=True)
t_2, N_2 = np.genfromtxt("data/WerteB.txt", unpack=True)
N_1 = N_1 - (N_u * 240)
N_2 = N_2 - (N_u * 10)
sq_N_1 = N_1**0.5
sq_N_2 = N_2**0.5


x = range(len(N_1))
y = range(len(N_2))

l_N_1 = np.log(N_1)
l_N_2 = np.log(N_2)



p_l_N_1 = np.log(N_1 + sq_N_1) - np.log(N_1)
m_l_N_1 = np.log(N_1) - np.log(N_1 - sq_N_1)

p_l_N_2 = np.log(N_2 + sq_N_2) - np.log(N_2)
m_l_N_2 = np.log(N_2) - np.log(N_2 - sq_N_2)

t_1, N_1 = np.genfromtxt("data/WerteA.txt", unpack=True)
t_2, N_2 = np.genfromtxt("data/WerteB.txt", unpack=True)
N_1 = N_1 - (N_u * 240)
N_2 = N_2 - (N_u * 10)


for i in x:
    print(t_1[i],"&",N_1[i],"&",sq_N_1[i], "\\\\")

for i in y:
    print(t_2[i],"&",N_2[i],"&",sq_N_2[i], "\\\\")

for i in x:
    print(t_1[i],"&", l_N_1[i], "&",p_l_N_1[i],"&",m_l_N_1[i], "\\\\" )

for i in y:
    print(t_2[i],"&", l_N_2[i], "&",p_l_N_2[i],"&",m_l_N_2[i], "\\\\" )


x = np.linspace(0, 3600, 10000)
params1, pcov1 = curve_fit(f, t_1, l_N_1)
print("Steigung1: ", params1[0] ,"+-", pcov1[0][0], " ln(N(0)(1-exp(-n dt))): ", params1[1],"+-", pcov1[1][1])
steigung1 = ufloat(params1[0], pcov1[0][0])
T = math.log(2)/steigung1
T_theo = 3257.4
dif_T = (T_theo - T.nominal_value)/T_theo
print("Halbwertszeit Indium: ", T, " s")
print("Theoriewert: ", T_theo)
print("Abw. Halbwertszeit lang: ", dif_T)

plt.errorbar(t_1, l_N_1,yerr=(m_l_N_1,p_l_N_1), fmt='k.', label='Messwerte')
plt.plot(x, f(x, *params1), 'r', label="Fit")
plt.grid()
plt.legend()
plt.xlabel("t in s")
plt.ylabel("ln(N)")
plt.savefig("build/plot1.pdf")
plt.clf()


y = np.linspace(100,420,1000)
z = np.linspace(0,100,1000)
yz = np.linspace(10,420,1000)
zy = np.linspace(0, 120, 1000)

params3, pcov3 = curve_fit(f,t_2[11:], l_N_2[11:])
print("Steigung3: ", params3[0] ,"+-", pcov3[0][0], " ln(N(0)(1-exp(-n dt))): ", params3[1],"+-", pcov3[1][1])
steigung3 = ufloat(params3[0], pcov3[0][0])
abschnitt3 = ufloat(params3[1],pcov3[1][1])
T_2_lang = np.log(2)/steigung3
T_2_lang_theo = 142.2
dif_T_2_lang = (T_2_lang_theo - T_2_lang.nominal_value)/T_2_lang_theo
print("Halbwertszeit lang: ", T_2_lang)
print("Theoriewert: ", T_2_lang_theo)
print("Abw. Halbwertszeit lang: ", dif_T_2_lang)

N_2_kurz = N_2[0:7]-np.exp(params3[1])*np.exp(-params3[0] * t_2[0:7]) 
sq_N_2_kurz = (N_2_kurz)**0.5
p_l_N_2_kurz = np.log(N_2_kurz + sq_N_2_kurz) - np.log(N_2_kurz)
m_l_N_2_kurz = np.log(N_2_kurz) - np.log(N_2_kurz - sq_N_2_kurz)
l_N_2_kurz = np.log(N_2_kurz)

k = range(len(N_2_kurz))
print("k")
for i in k:
    print(t_2[i],"&",N_2_kurz[i],"&",sq_N_2_kurz[i], "\\\\")

for i in k:
    print(t_2[i],"&", l_N_2[i], "&",p_l_N_2_kurz[i],"&",m_l_N_2_kurz[i], "\\\\" )

params2, pcov2 = curve_fit(f,t_2[0:7], l_N_2_kurz)
print("Steigung2: ", params2[0] ,"+-", pcov2[0][0], " ln(N(0)(1-exp(-n dt))): ", params2[1],"+-", pcov2[1][1])
steigung2 = ufloat(params2[0], pcov2[0][0])
T_2_kurz = np.log(2)/steigung2
T_2_kurz_theo = 24.6
dif_T_2_kurz = (T_2_kurz_theo - T_2_kurz.nominal_value)/T_2_kurz_theo
print("Halbwertszeit kurz: ", T_2_kurz)
print("Theoriewert: ", T_2_kurz_theo)
print("Abw. Halbwertszeit kurz: ", dif_T_2_kurz)

kombi_fkt = np.exp(params2[1]) * np.exp(-params2[0] * t_2) + np.exp(params3[1]) * np.exp(-params3[0] * t_2)
l_kombi_fkt = np.log(kombi_fkt)

y_g = [0,6]
x_g = [100,100]

plt.errorbar(t_2, l_N_2,yerr=(m_l_N_2,p_l_N_2), fmt='k.', label='Messwerte')
plt.plot(yz, f(yz, *params3), 'g', label="langlebig")
plt.plot(zy, f(zy, *params2), 'r', label="kurzlebig")
plt.plot(t_2, l_kombi_fkt, 'y', label="kombinerte Funktion")
plt.plot(x_g, y_g, "b--", label="t*")
plt.grid()
plt.legend()
plt.xlabel("t in s")
plt.ylabel("ln(N)")
plt.axis([0,430,0,5.8])
plt.savefig("build/plot3.pdf")
plt.clf()

plt.plot(z, f(z, *params2), 'r', label="kurzlebig")
plt.errorbar(t_2[:7], l_N_2_kurz,yerr=(m_l_N_2_kurz,p_l_N_2_kurz), fmt='k.', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel("t in s")
plt.ylabel("ln(N)")
plt.savefig("build/plot4.pdf")
plt.clf()

plt.plot(y, f(y, *params3), 'g', label="langlebig")
plt.errorbar(t_2[11:], l_N_2[11:],yerr=(m_l_N_2[11:],p_l_N_2[11:]), fmt='k.', label='Messwerte')
plt.grid()
plt.legend()
plt.xlabel("t in s")
plt.ylabel("ln(N)")
plt.savefig("build/plot5.pdf")
plt.clf()