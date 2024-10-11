import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from findiff import FinDiff
import matplotlib.pyplot as plt
""" 
Reference
https://medium.com/@mathcube7/two-lines-of-python-to-solve-the-schrödinger-equation-2bced55c2a0e
"""

x = np.linspace(-20, 20, 1001) # define our grid
dx = x[1]-x[0]

# d_dx = FinDiff(0, 1, 1)
# print(d_dx.matrix(x.shape).toarray())
d2_dx2 = FinDiff(0, dx, 2)

energies, states = eigs( -0.5 * d2_dx2.matrix(x.shape) + diags(0.5*x**2), k=61, which='SR')

print(d2_dx2.matrix(x.shape))

plt.plot(x, states[:, 0].real, label=r'$\psi_0$')
plt.plot(x, states[:, 1].real, label=r'$\psi_1$')
plt.plot(x, states[:, 2].real, label=r'$\psi_2$')
plt.plot(x, states[:, 60].real, label=r'$\psi_{60}$') # Compare to Figure 2.7 in Griffith
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
