import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon

f_e, U_br = np.genfromtxt('data\WerteE.txt', unpack=True) #Frequenz in Hz, Brückenspannung in mV
f_e, U1 = np.gentfromtxt('data\WerteE2.txt', unpack=True) #Frequenz in Hz, Signalspannung in mV
U_br = U_br * 1e3
U1 = U1 * 1e3