import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math

def T_a(t1,t2):
    return t2-t1

def U_a(U1,U2):
    return U2-U1

def e_fkt(t,A,B):
    return (A * np.exp(-B * t))

def hw(E,F):
    return E

# T_a/U_a ist die Steigung der einhüllenden Kurve

t_a, Spannung_a = np.genfromtxt('werteA.txt', unpack = True)
Frequenz_cd, Uc_c = np.genfromtxt('werteC.txt', unpack = True)
Frequenz_cd, delta_t_d = np.genfromtxt('werteD.txt', unpack =True)
t_a = t_a * 1e-6
delta_t_d = delta_t_d * 1e-6
Frequenz_cd = Frequenz_cd * 1e3

#Werte von Gerät 2
L = ufloat(10.11e-3, 0.03e-3) #in Henry
R1 = ufloat(48.1, 0.1)
R2 = ufloat(509.5, 0.5) #in Ohm
C = ufloat(2.098e-9, 0.006e-9) #in Farrad
R_ap = 3.28e3 #in Ohm
U_0 = 117 #in Volt
R_N = 50 #in Ohm    Der Wiederstand des Netzgeräts
w_0 = (1/(L * C))**(0.5) #Omega_Null
f_res = (1/(2 * math.pi)) * ((1/(L * C)) - ((R2+R_N)**2)/(4 * (L**2)))**(0.5) # errechnete Resonanz-Frequenz
#Teil a)

plt.plot(t_a, Spannung_a, label = 'Messdaten')
plt.xlabel('t in $\mu s$')
plt.ylabel('U in $V$')
#plt.xticks([0, 0 /  1,5e-6/ 35e-6/ 64e-6/ 94e-6/ 123e-6/ 152e-6/ 182e-6/ 212e-6/ 241e-6/ 270e-6/ 300e-6/ 330e-6],
 #           [0, 5, 35, 54, 94, 123, 152, 182, 212, 241,270,300, 330])
params, cov_matrix = curve_fit(e_fkt, t_a, Spannung_a)
errors = np.sqrt(np.diag(cov_matrix))
print('A = ', params[0], '+/-', errors[0])  
print('B = ', params[1], '+/-', errors[1])  #mu
plt.plot(t_a, e_fkt(t_a, *params), 'r-', label = 'Exp. Fit')
plt.legend()
plt.savefig('build\plot1.pdf')
mu = ufloat(params[1], errors[1]) / (2 * math.pi)           #mu
plt.clf()

T_ex_versuch = 1 / (2 * math.pi * mu)
T_ex_theorie = 2 * L / (R2+R_N)
T_ex_prozent = T_ex_versuch / T_ex_theorie

print('T_ex_versuch = ', T_ex_versuch,'T_ex_theorie = ', T_ex_theorie, 'T_ex_prozent = ', T_ex_prozent)

R_eff = 4 * math.pi * mu * L
print('R_eff = ',R_eff)
print('mu = ', mu)

#Teil c)

R_ap_rechn=(4 * L / C)**(0.5)
R_ap_prozent = R_ap / R_ap_rechn
print('R_ap = ', R_ap, 'R_ap_rechn = ', R_ap_rechn, 'R_ap_prozent = ', R_ap_prozent) #Prozentuale Abweichung vom Gemessenen zum Errechneten Wert
U_diff = [x / U_0 for x in Uc_c]
plt.plot(Frequenz_cd, U_diff, 'k.', label = 'Messdaten')
plt.xlabel('f in $kHz$')
plt.ylabel('$U$ / $U_0$')
plt.legend()
plt.savefig('build\plot2.pdf')
q_rechn = 1 / (w_0 * C * (R2+R_N)) #Resonanzüberhöhung, Güte

plt.clf()
print('U_diff         Frequenz_cd')
for x in range (0, 19):
    print(U_diff[x], Frequenz_cd[x])

f_res_gemessen = 33 #in kHz, Frequenz zum Maximum aus Plot2
q_gemessen = 2.45 #Maximum vom Plot2
q_halbw = 1/(np.sqrt(2)) * q_gemessen
q_prozent = q_gemessen / q_rechn
print('q_rechn = ', q_rechn, 'q_gemessen = ',q_gemessen, 'q_prozent = ', q_prozent )
print('q_halbw = ', q_halbw)
plt.plot(Frequenz_cd, U_diff, label = 'Messdaten')
plt.xlabel('f in $kHz$')
plt.ylabel('$U$ / $U_0$')
plt.xlim(25000, 40000)
plt.ylim(1.4, 2.5)
q_array = np.genfromtxt('q_halbw.txt', unpack = True)
plt.plot(Frequenz_cd, q_array, 'r--')
f_array = np.genfromtxt('f_halbw.txt', unpack = True) #27.20 kHz
f_array2 = np.genfromtxt('f_halbw2.txt', unpack = True) #38.34 kHz
y_gerade = np.genfromtxt('y_werte.txt', unpack = True)
plt.plot(f_array,y_gerade,'r--', label = 'Halbwertsbreite')
plt.plot(f_array2,y_gerade,'r--')
plt.legend()

plt.savefig('build\plot3.pdf')

plt.clf()

nu1 = (-(R2+R_N)/(2*L) + (((R2+R_N)**2/(4 * L**2)) + (1/(L * C)))**(0.5)) / (2 * math.pi)
nu2 = ((R2+R_N)/(2*L) + ((R2+R_N)**2/(4 * L**2) + 1/(L * C))**(0.5)) / (2 * math.pi)

print('nu1 = ', nu1, 'nu2 = ', nu2)


#Teil d)

phi = delta_t_d * Frequenz_cd *2 * math.pi

print('phi = ', phi)

f_array_res = np.genfromtxt('f_res.txt', unpack = True)

plt.plot(Frequenz_cd, phi, 'k.', label = 'Messdaten')
plt.xlabel('f in $kHz$')
plt.ylabel('$U$ / $U_0$')
plt.plot(f_array,y_gerade,'r--', label = r'$\nu_+$ bzw. $\nu_-$')
plt.plot(f_array2,y_gerade,'r--')
plt.plot(f_array_res, y_gerade, 'g--', label = 'Resonanzfrequenz')
plt.legend()
plt.savefig('build\plot4.pdf')