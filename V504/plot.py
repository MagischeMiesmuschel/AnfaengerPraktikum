import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon

x = np.linspace(0,10,10)
plt.plot(x,x**2)
plt.savefig('build/plot1.pdf')
