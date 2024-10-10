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
speed_of_light = 3*10**8

well_height = 0.3 #in eV
well_height = well_height/27.211 # in a.u.

well_width =  3 # in nm
well_width = int(well_width/0.005291772) # in a.u.

wall_width = 1
wall_width = int(wall_width/0.005291772) # in a.u.

side_buffer = int(well_width/10)
half_domain = well_width + 0.5*wall_width + side_buffer

x_ticks = 3000
tick_width = half_domain/(x_ticks/2)

start_buffer_ticks = int(side_buffer/tick_width)
well_ticks = int(well_width/tick_width)
wall_ticks = int(wall_width/tick_width)
end_buffer_ticks = x_ticks - (start_buffer_ticks + 2*well_ticks + wall_ticks)

x = np.linspace(-half_domain, half_domain, x_ticks) # define our grid
dx = x[1]-x[0]

potential_vector = well_height*np.concatenate([np.ones(start_buffer_ticks), np.zeros(well_ticks), np.ones(wall_ticks), np.zeros(well_ticks), np.ones(end_buffer_ticks)])

plt.plot(x,potential_vector)
plt.show()
potential_matrix = diags(potential_vector)

d2_dx2 = FinDiff(0, dx, 2)

operator = -0.5 * d2_dx2.matrix(x.shape) + potential_matrix
energies, states = eigs( operator, k=4, which='SR')



plt.plot(x, np.abs(states[:, 0])**2, label=r'$\psi_0$')
plt.plot(x, np.abs(states[:, 1])**2, label=r'$\psi_1$')
plt.plot(x, np.abs(states[:, 2])**2, label=r'$\psi_2$')
plt.plot(x, np.abs(states[:, 3])**2, label=r'$\psi_3$')
plt.grid()
plt.legend()
plt.xlim(min(x), max(x))
plt.show()


# fig = plt.figure(figsize=(5,8))
# ax = fig.gca()
# levels = [[(0, 1), (e.real, e.real)] for e in energies]
# for level in levels[:5]:    
#     ax.plot(level[0], level[1], '-b')
# ax.set_xticks([])
# ax.set_ylabel('energy [a.u.]')
# plt.show()
