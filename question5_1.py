import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt

""" 
Reference
https://medium.com/@mathcube7/two-lines-of-python-to-solve-the-schrödinger-equation-2bced55c2a0e
"""

planck_constant = 4.135668*10**(-15) # in eV
speed_of_light = 299792458

well_depth = 1 #in eV
well_depth = well_depth/27.211386245988 # in a.u.

well_width_average =  3 # in nm

width_linspace = np.linspace(well_width_average - 1, well_width_average + 1, 6)

n_energies = 30
energy_levels = dict({})

for well_width in width_linspace:
    print(f"Iteration avec {well_width} nm ...")
    well_width = well_width/0.005291772 # in a.u.

    x = np.linspace(-well_width, well_width, 1001) # define our grid
    dx = x[1]-x[0]

    potential_vector = np.where(np.abs(x)<=(well_width/2), 0, well_depth)

    potential_matrix = diags(potential_vector)

    d2_dx2 = FinDiff(0, dx, 2)

    operator = -0.5 * d2_dx2.matrix(x.shape) + potential_matrix
    energies, states = eigs( operator, k=n_energies, which='SR')

    five_energies = energies.real

    energy_levels[well_width * 0.005291772] = five_energies
    print("done")
    
print(energy_levels)

for width, energies_array in energy_levels.items():
    plt.scatter([i + 1 for i in range(n_energies)], energies_array, label=r'$d = $' + f"{width:.2f} nm")

plt.grid()
plt.legend()
plt.xlim(0, n_energies + 1)
plt.xlabel(r"Niveau d'énergie $n$", fontsize=20)
plt.ylabel(r'Énergie [u.a.]', fontsize=20)
plt.show()


