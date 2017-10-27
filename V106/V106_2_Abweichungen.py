import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

NormalLinks1 = np.array([2.00, 1.93, 2.07, 2.00, 1.92, 2.04, 1.95, 1.96, 1.94, 1.90])
NormalLinks2 = np.array([1.96, 2.02, 1.93, 2.06, 1.93, 2.03, 2.06, 1.95, 1.97, 1.96])
NormalRechts1 = np.array([2.01, 2.02, 1.96, 2.00, 1.94, 2.00, 1.90, 2.03, 1.96, 2.01 ])
NormalRechts2 = np.array([2.10, 1.92, 1.97, 2.02, 1.93, 1.96, 2.00, 1.96, 2.02, 1.96])
GlLinks = np.array([1.94, 1.96, 1.99, 2.03, 1.99, 1.93, 2.04, 2.03, 1.96, 2.00])
GlRechts = np.array([1.90, 2.02, 2.10, 1.90, 2.06, 1.93, 1.95, 1.97, 2.00, 2.04])
GegLinks = np.array([1.49, 1.48, 1.48, 1.43, 1.38, 1.43, 1.52, 1.40, 1.43, 1.40])
GegRechts = np.array([1.48, 1.43, 1.35, 1.50, 1.47, 1.30, 1.40, 1.46, 1.52, 1.43])
GekLinks = np.array([2.03, 2.05, 1.89, 2.07, 1.83, 2.13, 1.93, 1.86, 2.13, 1.80])
GekRechts = np.array([1.89, 2.10, 1.85, 2.11, 1.96, 1.89, 2.10, 1.92, 2.10, 1.90])
Schwebung = np.array([4.90, 5.13, 5.54, 4.48, 5.07, 4.94, 5.01, 5.17, 4.98, 5.03])

print('Pendellänge von 100cm, Auslenkungen von 7cm, Mit Kleinwinkel bis zu 17,36cm')
print('')

print('Schwingungsdauer, Pendel links1 ohne Feder')
print(np.mean(NormalLinks1))
print(np.std(NormalLinks1))

print('Schwingungsdauer, Pendel links2 ohne Feder')
print(np.mean(NormalLinks2))
print(np.std(NormalLinks2))

print('Schwingungsdauer, Pendel rechts1 ohne Feder')
print(np.mean(NormalRechts1))
print(np.std(NormalRechts1))

print('Schwingungsdauer, Pendel rechts2 ohne Feder')
print(np.mean(NormalRechts2))
print(np.std(NormalRechts2))

print('Schwingungsdauer, Pendel links Gleichsinnig')
print(np.mean(GlLinks))
print(np.std(GlLinks))

print('Schwingungsdauer, Pendel rechts Gleichsinnig')
print(np.mean(GlRechts))
print(np.std(GlRechts))

print('Schwingungsdauer, Pendel links Gegensinnig')
print(np.mean(GegLinks))
print(np.std(GegLinks))

print('Schwingungsdauer, Pendel rechts Gegensinnig')
print(np.mean(GegRechts))
print(np.std(GegRechts))

print('Schwingungsdauer, Pendel links Gekoppelt')
print(np.mean(GekLinks))
print(np.std(GekLinks))

print('Schwingungsdauer, Pendel rechts Gekoppelt')
print(np.mean(GekRechts))
print(np.std(GekRechts))

print('Schwingungsdauer, Pendel Schwebung')
print(np.mean(Schwebung))
print(np.std(Schwebung))

