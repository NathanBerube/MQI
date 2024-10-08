import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt

""" 
Reference
https://medium.com/@mathcube7/two-lines-of-python-to-solve-the-schrödinger-equation-2bced55c2a0e
"""
planck_constant = 4.135668*10**(-15)
speed_of_light = 3*10**8
m_e = 9.1 * 10**(-31)

hamiltonian_constant = -planck_constant**2/(2*m_e)

well_depth = 1 #in eV
well_width_guess = 42

wavelength = 10.6 # in µm

iterations = 20
optimal_width = well_width_guess
optimal_wavelength = 0
difference_with_target = wavelength


for well_width in np.linspace(well_width_guess - 0.5, well_width_guess + 0.5, iterations):
    x = np.linspace(-well_width, well_width, 1001) # define our grid
    dx = x[1]-x[0]

    potential_vector = np.where(np.abs(x)<=(well_width/2), 0, well_depth)

    potential_matrix = diags(potential_vector)

    d2_dx2 = FinDiff(0, dx, 2)

    operator = hamiltonian_constant * d2_dx2.matrix(x.shape) + potential_matrix
    energies, states = eigs( operator, k=2, which='SR')


    energy_diff_1_2 = energies[1] - energies[0]

    current_wavelength = ((planck_constant * speed_of_light)/energy_diff_1_2) * 10**6

    if current_wavelength > wavelength + 1:
        break

    current_difference = abs(wavelength - current_wavelength)
    print(f"Current well width {well_width}")
    print(f"Current wavelength {current_wavelength}")
    print("\n")
    if current_difference < difference_with_target:
        difference_with_target = current_difference
        optimal_width = well_width
        optimal_wavelength = current_wavelength

print(f"Optimal well width is: {optimal_width}")
print(f"Optimal emitted wavelength is: {optimal_wavelength}")

# plt.plot(x, states[:, 0].real, label=r'$\psi_0$')
# plt.plot(x, states[:, 1].real, label=r'$\psi_1$')
# # plt.plot(x, states[:, 2].real, label=r'$\psi_2$')
# plt.grid()
# plt.legend()
# plt.xlim(min(x), max(x))
# plt.show()


# fig = plt.figure(figsize=(5,8))
# ax = fig.gca()
# levels = [[(0, 1), (e.real, e.real)] for e in energies]
# for level in levels[:5]:    
#     ax.plot(level[0], level[1], '-b')
# ax.set_xticks([])
# ax.set_ylabel('energy [a.u.]')
# plt.show()
