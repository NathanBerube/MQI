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

well_width =  3 # in nm
well_width =  3/0.005291772  # in a.u.
barrier_width_average = 3 # in nm

barrier_linspace = np.linspace(barrier_width_average - 2.5, barrier_width_average - 2.4, 20)
n_modes = 5

prob_crosstalk = dict({})



x = np.linspace(-5 * well_width, 5 * well_width, 3001) # define our grid
dx = x[1]-x[0]

potential_vector = np.where(np.abs(x)<=(well_width/2), 0, well_depth)
# plt.plot(x, potential_vector)
potential_matrix = diags(potential_vector)

d2_dx2 = FinDiff(0, dx, 2)

operator = -0.5 * d2_dx2.matrix(x.shape) + potential_matrix
energies, states = eigs( operator, k=n_modes, which='SR')

plt.plot(x, states[:, 0].real, label=r'$\psi_0$')
plt.plot(x, states[:, 1].real, label=r'$\psi_1$')
plt.plot(x, states[:, 2].real, label=r'$\psi_2$')
plt.show()
for barrier_width in barrier_linspace:
    barrier_width_au = barrier_width/0.005291772 # in a.u.
    print(f"Iteration avec {barrier_width} nm ...")

    start_pos_integration = well_width/2 + barrier_width_au

    index_pos_integration = np.argmin(np.abs(x - start_pos_integration))

    prob = []
    for i in range(n_modes):
        state = states[:, i].real
        prob_density = np.abs(state)**2
        prob_beyond = np.sum(dx * prob_density[index_pos_integration:])
        prob.append(prob_beyond)


    prob_crosstalk[barrier_width] = prob

for n in range(n_modes):
    prob_at_n = []
    for barrier_width in barrier_linspace:
        prob_array = prob_crosstalk[barrier_width]
        prob_at_mode = prob_array[n]
        prob_at_n.append(prob_at_mode)
    plt.scatter(barrier_linspace, prob_at_n, label=f"E_{n+1}")

plt.grid()
plt.legend()
# plt.ylim()
# plt.xlim(barrier_width_average - 0.1, barrier_width_average + 0.1)

plt.yscale("log")
plt.xlabel(r"Épaisseur $d$", fontsize=20)
plt.ylabel(r'Probabilité [-]', fontsize=20)
plt.show()


