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


well_width = 50.95 # in a.u.




x = np.linspace(-well_width, well_width, 1001) # define our grid
dx = x[1]-x[0]

potential_vector = np.where(np.abs(x)<=(well_width/2), 0, well_depth)

potential_matrix = diags(potential_vector)

d2_dx2 = FinDiff(0, dx, 2)

operator = -0.5 * d2_dx2.matrix(x.shape) + potential_matrix
energies, states = eigs( operator, k=3, which='SR')

energy_diff_1_2 = energies[1] - energies[0] # in a.u.
energy_diff_1_2 = energy_diff_1_2 * 27.211386245988 # in eV

wavelength =  (((planck_constant * speed_of_light)/energy_diff_1_2) * 10**6).real # in µm

well_width_si = well_width * 0.5291772105

print(f"Wavelength is: {wavelength} µm")
print(f"Well width is: {well_width_si} angstrom")


plt.plot(x, states[:, 0].real, label=r'$\psi_0$')
plt.plot(x, states[:, 1].real, label=r'$\psi_1$')
plt.plot(x, states[:, 2].real, label=r'$\psi_2$')
plt.grid()
plt.legend()
plt.xlim(min(x), max(x))
plt.show()


fig = plt.figure(figsize=(5,8))
ax = fig.gca()
levels = [[(0, 1), (e.real, e.real)] for e in energies]
for level in levels[:5]:    
    ax.plot(level[0], level[1], '-b')
ax.set_xticks([])
ax.set_ylabel('energy [a.u.]')
plt.show()
