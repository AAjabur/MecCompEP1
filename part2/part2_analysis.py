from utils.gauss_seidel import relaxation_gauss_seidel
from utils.mdf_equation_generator import MdfPsiEquationGenerator
import matplotlib.pyplot as plt
import numpy as np

psi_eq_gen = MdfPsiEquationGenerator(delta=0.1)

x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

init_psi_guess = psi_eq_gen.generate_initial_psi_matrix()

iterate_func = psi_eq_gen.iterate_psi

psi_matrix = relaxation_gauss_seidel(iterate_func, init_psi_guess, goal_relative_error=0.01)

print(psi_matrix)

fig, ax = plt.subplots(1,1)
cp = ax.contour(x_matrix, y_matrix, psi_matrix, 20)
ax.clabel(cp, inline=True, fontsize=10)

plt.show()
