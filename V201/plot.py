import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import math

def m(x,y):
    z=0
    for a in range(0,y):
        z = z + (x[a] - np.mean(x))**2
    return 1/math.sqrt(y) * math.sqrt(1/(y-1)*z)

#Theoriewerte
c_blei_theo = 0.130
c_kupfer_theo = 0.385
c_alu_theo = 0.897
C_dulong = 24.9

#Teil 1 c von Kalo bestimmen
m_kalo = 841.85
c_w = 4.18    #Joul / g * K     also alles in Gramm!
m_wk = 312.42
m_ww = 300.54
T_wk = 20.6
T_ww = 86
T_wm = 50

c_kalo = (c_w * m_ww * (T_ww - T_wm) - c_w * m_wk * (T_wm - T_wk)) / ((T_wm - T_wk) * m_kalo)
print('c_kalo = ', c_kalo)

#Teil 2 c von Blei
m_blei = 535.33
m_w1,T_w1,T_blei,T_m1 = np.genfromtxt('data\Werte_blei.txt', unpack=True)
c_blei = ((c_w * m_w1 + c_kalo * m_kalo)*(T_m1 - T_w1))/(m_blei * (T_blei - T_m1))
c_blei_d = np.mean(c_blei)
print('c_blei = ', c_blei)
print('c_blei Durchschnitt = ', c_blei_d , ' +/- ', m(c_blei, 3))
print('Prozentuale Abweichung von der Theorie : ', 100*(c_blei_d - c_blei_theo)/c_blei_theo)
C_blei_mol = c_blei * 207.2
C_blei_mol_d = np.mean(C_blei_mol)
print('C_blei_mol = ', C_blei_mol_d, ' +/- ', m(C_blei_mol,3))
print('Prozentuale Abweichung von der Theorie : ',100*(C_blei_mol_d - C_dulong)/C_dulong)

#Teil 2 c von Kupfer
m_kupfer = 230.2
m_w2,T_w2,T_kupfer,T_m2 = np.genfromtxt('data\Werte_kupfer.txt', unpack=True)
c_kupfer = ((c_w * m_w2 + c_kalo * m_kalo)*(T_m2 - T_w2))/(m_kupfer * (T_kupfer - T_m2))
c_kupfer_d = np.mean(c_kupfer)
print('c_kupfer = ', c_kupfer)
print('c_kupfer Durchschnitt = ', c_kupfer_d, ' +/- ', m(c_kupfer, 3))
print('Prozentuale Abweichung von der Theorie : ', 100*(c_kupfer_d - c_kupfer_theo)/c_kupfer_theo)
C_kupfer_mol = c_blei * 63.5
C_kupfer_mol_d = np.mean(C_kupfer_mol)
print('C_kupfer_mol = ', C_kupfer_mol_d, ' +/- ', m(C_kupfer_mol,3))
print('Prozentuale Abweichung von der Theorie : ',100*(C_kupfer_mol_d - C_dulong)/C_dulong)

#Teil 2 c von Alu
m_alu = 149.03
m_w3,T_w3,T_alu,T_m3 = np.genfromtxt('data\Werte_alu.txt', unpack=True)
c_alu = ((c_w * m_w3 + c_kalo * m_kalo)*(T_m3 - T_w3))/(m_alu * (T_alu - T_m3))
c_alu_d =np.mean(c_alu)
print('c_alu = ', c_alu)
print('c_alu Durchschnitt = ',c_alu_d , ' +/- ', m(c_alu, 3))
print('Prozentuale Abweichung von der Theorie : ', 100*(c_alu_d - c_alu_theo)/c_alu_theo)
C_alu_mol = c_blei * 27
C_alu_mol_d = np.mean(C_alu_mol)
print('C_alu_mol = ', C_alu_mol_d, ' +/- ', m(C_alu_mol,3))
print('Prozentuale Abweichung von der Theorie : ',100*(C_alu_mol_d - C_dulong)/C_dulong)