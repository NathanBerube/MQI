import numpy as np
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt

# Largeur du puits de potentiel (en unités atomiques)
L = 56.6918  # L en a0 (longueur de Bohr)

# Profondeur du puits de potentiel (en unités atomiques, 1 UA = 27.2114 eV)
V = 3*0.0367502  # V0 en Hartree (par exemple, 0.5 Hartree)

# Définir la cotangente
def cot(x):
    return 1 / np.tan(x)

# Fonctions transcendantes pour les états pairs
def eq_pair(E):
    k = np.sqrt(2 * E)
    kappa = np.sqrt(2 * (V - E))
    return k * cot(k * L / 2) + kappa

# Fonctions transcendantes pour les états impairs
def eq_impair(E):
    k = np.sqrt(2 * E)
    kappa = np.sqrt(2 * (V - E))
    return k * np.tan(k * L / 2) - kappa

# Deviner des valeurs initiales pour E (en Hartree)
# initial_guesses_impair = [0.0013, 0.012, 0.033, 0.064, 0.103]
# initial_guesses_pair = [0.0053, 0.021, 0.047, 0.083]
initial_guesses = [0.0013, 0.0053, 0.012, 0.021, 0.033, 0.047, 0.064, 0.083, 0.103]
energies = []
T = []
barriers = np.linspace(0, 5, 5001)

n = 1
for guess in initial_guesses:
    if (n % 2) == 1:
        # print(f"{n} is odd")
        energies.append(fsolve(eq_impair, guess)[0])
    else:
        energies.append(fsolve(eq_pair, guess)[0])

    t = []

    for barrier in barriers:
        t.append((1 + V**2/(4*energies[-1]*(V - energies[-1])) * (math.sinh(barrier*18.8973*math.sqrt(2*(V - energies[-1]))))**2)**-1 * 100)

    plt.plot(barriers, t, label=r'$E = $' + f"{energies[-1]*27.2114:.3f} eV")

    n += 1



# for barrier in barriers:
#     t = []

#     for E in energies:
#         t.append((1 + V**2/(4*E*(V - E)) * (math.sinh(barrier*18.8973*math.sqrt(2*(V - E))))**2)**-1 * 100)

#     T.append(t)
#     plt.scatter(energies, t, label=r'$d = $' + f"{barrier} nm")

# l = 0

# for E in energies:
#         energies[l] = E*27.2114
#         l += 1

# print(energies)
# print(T)

# for t in T:
#     plt.scatter(energies, t, label=r'$d = $' + f"{width:.2f} nm")
    
# plt.grid()
plt.legend()
plt.yscale("log")
plt.xlabel(r"Épaisseur de la barrière [nm]", fontsize=14)
plt.ylabel(r'Transmission [%]', fontsize=14)
plt.show()

# n = 1
# # Résoudre les équations
# for k in range(len(initial_guesses_impair)):
#     # États impairs
#     E_impair_solution = fsolve(eq_impair, initial_guesses_impair[k])
    
#     # Afficher les résultats (en Hartree)
#     print(f"Niveau d'énergie impair {n} : {E_impair_solution[0]:.5f} Hartree")

#     n += 1

# n = 1
# # Résoudre les équations
# for k in range(len(initial_guesses_pair)):
#     # États pairs
#     E_pair_solution = fsolve(eq_pair, initial_guesses_pair[k])
    
#     # Afficher les résultats (en Hartree)
#     print(f"Niveau d'énergie pair {n} : {E_pair_solution[0]:.5f} Hartree")

#     n += 1
