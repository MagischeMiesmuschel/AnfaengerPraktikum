import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math

t_o, t_u, s_o, s_u = np.genfromtxt("data/werte.txt", unpack=True)
h = 80.3e-3
c = 2730
c_w = 1480

t_1 = 45.8e-6
t_2 = 23.9e-6 
d_herz = 45.4e-3

x = range(len(t_o))

for i in x:
    print(i+1 , " & ", t_o[i], " & ", t_u[i], " & ", s_o[i] , " & ", s_u[i], " \\\\")

t_o = t_o*10**(-6)
t_u = t_u*10**(-6)
s_o = s_o*10**(-3)
s_u = s_u*10**(-3)

d_o = (c * (t_o/2) )* 10**(3) # in mm
d_u = (c * (t_u/2) )* 10**(3) # in mm
d = h*10**3 - d_o - d_u # in mm
d_mess = (h - s_o - s_u)*10**3
d_abw = (d-d_mess)/d_mess *100

for i in x:
    print(i+1 , " & ",d_o[i], " & ", d_u[i], "&", d[i], "&", d_mess[i], "&", d_abw[i], " \\\\")

s_h1 = c_w * (t_1/2) 
s_h2 = c_w * (t_2/2) 
V_h1 = s_h1 * math.pi * (d_herz/2)**2
V_h2 = s_h2 * math.pi * (d_herz/2)**2
s_h1 = s_h1 * 10**(3) # in mm
s_h2 = s_h2 * 10**(3) # in mm
f_h = 9/19 #9 Schwingungen in 19 Sekunden
V_h = (V_h1-V_h2)*f_h

print("f_h: ", f_h, " Hz")
print("V_h1: ", V_h1, " m^3")
print("V_h2: ", V_h2, " m^3")
print("V_h: ", V_h, " m^3")
print("s_h1: ", s_h1, " mm")
print("s_h2: ", s_h2, " mm")