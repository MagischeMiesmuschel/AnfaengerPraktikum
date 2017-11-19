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


# T_a/U_a ist die Steigung der einhüllenden Kurve

t_a = [5, 35, 64, 94, 123, 152, 182, 212, 241, 270, 300, 330, 359, 388] #in Mikro Sekunden
Spannung_a = [235, 217, 202, 188, 177, 166, 160, 152, 147, 142, 138, 135, 132, 130] #in Volt
Frequenz_cd = [15, 20, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 45, 50] #in Kilo Herz
Uc_c = [130, 146, 178, 189, 200, 214, 229, 247, 264, 279, 287, 285, 272, 253, 230, 210, 192, 176, 126, 102] #in V
delta_t_d = [1.2, 1.6, 2.4, 2.8, 3.2, 3.6, 3.8, 4.4, 5.0, 5.6, 6.8, 7.6, 8.4, 8.8, 9.4, 9.6, 10, 10, 9.8, 9.2] #in Mikro Sekunden

#Werte von Gerät 2
L = ufloat(10.11e-3, 0.03e-3) #in Henry
R1 = ufloat(48.1, 0.1)
R2 = ufloat(509.5, 0.5) #in Ohm
C = ufloat(2.098e-9, 0.006e-9) #in Farrad
R_ap = 3.28e3 #in Ohm
U_0 = 117 #in Volt
R_N = 50 #in Ohm    Der Wiederstand des Netzgeräts
w_0 = (1/(L * C))**(0.5) #Omega_Null
f_res = 1/(2 * math.pi) * ((1/(L * C)) - ((R2+R_N)**2)/2 * (L**2))**(0.5) # errechnete Resonanz-Frequenz
print(f_res) 
plt.plot(t_a, Spannung_a)
plt.xlabel('t in $\mu s$')
plt.ylabel('U in $V$')
plt.savefig('plot1.pdf')

R_ap_rechn=(4 * L / C)**(0.5)
R_ap_prozent = R_ap / R_ap_rechn
print(R_ap_prozent) #Prozentuale Abweichung vom Gemessenen zum Errechneten Wert
U_diff = [x / U_0 for x in Uc_c]
plt.plot(Frequenz_cd, U_diff)
plt.xlabel('F in $kHz$')
plt.ylabel('$U$ / $U_0$')
plt.savefig('plot2.pdf')
q = 1 / (w_0 * C * R2) #Resonanzüberhöhung, Güte
print(q)
for x in range (0, 19):
    print(U_diff[x])