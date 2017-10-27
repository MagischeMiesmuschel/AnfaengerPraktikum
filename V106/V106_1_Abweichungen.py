import numpy as np

NormalLinks1 = np.array([1.56, 1.43, 1.43, 1.47, 1.41, 1.56, 1.44, 1.43, 1.40, 1.42])
NormalLinks2 = np.array([1.50, 1.43, 1.38, 1.49, 1.46, 1.47, 1.45, 1.47, 1.45, 1.42])
NormalRechts1 = np.array([1.49, 1.33, 1.43, 1.60, 1.49, 1.40, 1.45, 1.47, 1.50, 1.43])
NormalRechts2 = np.array([1.45, 1.46, 1.43, 1.44, 1.46, 1.42, 1.47, 1.39, 1.50, 1.39])
GlLinks = np.array([1.51, 1.42, 1.50, 1.43, 1.46, 1.46, 1.43, 1.45, 1.50, 1.39])
GlRechts = np.array([1.42, 1.43, 1.46, 1.43, 1.49, 1.50, 1.37, 1.53, 1.32, 1.54])
GegLinks = np.array([0.90, 1.06, 1.00, 0.82, 1.13, 0.90, 1.02, 0.93, 1.04, 0.83])
GegRechts = np.array([0.96, 0.96, 1.03, 0.89, 1.00, 0.96, 0.99, 0.96, 0.94, 0.92])
GekLinks = np.array([1.60, 1.36, 1.52, 1.54, 1.32, 1.60, 1.30, 1.50, 1.53, 1.40])
GekRechts = np.array([1.42, 1.43, 1.53, 1.34, 1.53, 1.42, 1.57, 1.47, 1.50, 1.49])
Schwebung = np.array([2.99, 2.80, 2.82, 2.94, 2.90, 2.90, 2.73, 2.80, 2.91, 2.83])

print('Pendellänge von 50cm, Auslenkungen von 5cm, Mit Kleinwinkel bis zu 8,68cm')
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

