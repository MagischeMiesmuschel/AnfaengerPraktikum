from uncertainties import ufloat
import numpy as np

TGl1 = 0.5 * (ufloat(1.455, 0.037) + ufloat(1.449, 0.066))
TGeg1 = 0.5 * (ufloat(0.963, 0.097) + ufloat(0.961, 0.038))
TGl2 = 0.5 * (ufloat(1.987, 0.037) + ufloat(1.987, 0.065))
TGeg2 = 0.5 * (ufloat(1.444, 0.044) + ufloat(1.434, 0.065))

print(TGl1)
print(TGl2)
print(TGeg1)
print(TGeg2)

WGl1 = (2 * np.pi) / TGl1
WGl2 = (2 * np.pi) / TGl2
WGeg1 = (2 * np.pi) / TGeg1
WGeg2 = (2 * np.pi) / TGeg2

K1 = (WGeg1**2 - WGl1**2) / (WGeg1**2 + WGl1**2)
K2 = (WGeg2**2 - WGl2**2) / (WGeg2**2 + WGl2**2)
K = 0.5 * (K1 + K2)
print(WGl1, WGeg1, WGl2, WGeg2)
print(K1, K2, K)

TS1 = (TGl1 * TGeg1) / (TGl1 - TGeg1)
TS2 = (TGl2 * TGeg2) / (TGl2 - TGeg2)

print(TS1, TS2)

WGegr1 = (9.81 / 0.5)**0.5 * ((1+K)/(1-K))**0.5
WGegr2 = (9.81 / 1)**0.5 * ((1+K)/(1-K))**0.5

print(WGegr1, WGegr2)

TGegr1 = 2 * np.pi / WGegr1
TGegr2 = 2 * np.pi / WGegr2

print(TGegr1, TGegr2)