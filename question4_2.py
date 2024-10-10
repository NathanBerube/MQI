import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt

""" 
Reference
https://medium.com/@mathcube7/two-lines-of-python-to-solve-the-schrödinger-equation-2bced55c2a0e
"""

plt.rcParams['text.usetex'] = True

planck_constant = 4.135668*10**(-15) # in eV
speed_of_light = 3*10**8

well_depth = 1 #in eV
well_depth = well_depth/27.211 # in a.u.


well_width = 50.92 # in a.u., de la sous-question 1

n_time_ticks = 300
n_position_ticks = 1001
end_time = 0.0000000000003 # durée en temps du graphique de la valeur moyenne, faible, car fréquence très élevée


time_array = np.linspace(0, end_time, n_time_ticks)

x = np.linspace(-well_width, well_width, n_position_ticks) # define our grid

dx = x[1]-x[0]

potential_vector = np.where(np.abs(x)<=(well_width/2), 0, well_depth)

potential_matrix = diags(potential_vector)

d2_dx2 = FinDiff(0, dx, 2)

operator = -0.5 * d2_dx2.matrix(x.shape) + potential_matrix
energies, states = eigs( operator, k=3, which='SR')

energy_diff_0_1 = energies[1] - energies[0] # in a.u.
energy_diff_0_1 = energy_diff_0_1 * 27.211 # in eV

wavelength =  (((planck_constant * speed_of_light)/energy_diff_0_1) * 10**6).real # in µm

well_width_si = well_width * 0.529177

print(f"Wavelength is: {wavelength} µm")
print(f"Well width is: {well_width_si} angstrom")



def mode_n(n, t):
    energy = energies[n] * 21.211 # in eV
    omega = 2 * np.pi * energy/planck_constant
    mode = states[:,n] * np.exp(-1j * omega * t)
    return mode

mode_0_matrix = np.array([])
for time in time_array:
    mode_0_matrix = np.append(mode_0_matrix, mode_n(0,time))

mode_1_matrix = np.array([])
for time in time_array:
    mode_1_matrix = np.append(mode_1_matrix, mode_n(1,time))

mode_2_matrix = np.array([])
for time in time_array:
    mode_2_matrix = np.append(mode_2_matrix, mode_n(2,time))


mode_0_matrix = np.reshape(mode_0_matrix, (n_time_ticks, n_position_ticks))
mode_1_matrix = np.reshape(mode_1_matrix, (n_time_ticks, n_position_ticks))
mode_2_matrix = np.reshape(mode_2_matrix, (n_time_ticks, n_position_ticks))

print(x.shape)
def average_position_from_state_matrix(state):
    conjugate = np.conjugate(state)
    average_position = np.mean(conjugate * x * state, axis=1)
    # average_position = average_position.reshape((n_time_ticks,1))
    return average_position


state_0_1 = 1/np.sqrt(2) * (mode_0_matrix + mode_1_matrix)
state_0_2 = 1/np.sqrt(2) * (mode_0_matrix + mode_2_matrix)

average_position_0_1 = average_position_from_state_matrix(state_0_1)
average_position_0_2 = average_position_from_state_matrix(state_0_2)


plt.plot(time_array * 10**13, average_position_0_1.real, label=r'Superposition $\psi_1$ et $\psi_2$', color='black')
plt.plot(time_array * 10**13, average_position_0_2.real, label=r'Superposition $\psi_1$ et $\psi_3$', color='red')
plt.xlabel(r'Temps $t$ [$10^{-13}$ s]', fontsize=20)
plt.ylabel(r'Position moyenne $\langle x \rangle$ [u.a.]', fontsize=20)
plt.legend()
plt.show()