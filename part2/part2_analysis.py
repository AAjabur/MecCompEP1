from utils.gauss_seidel import relaxation_gauss_seidel
from utils.mdf_equation_generator import MdfPsiEquationGenerator
import matplotlib.pyplot as plt
import numpy as np

psi_eq_gen = MdfPsiEquationGenerator(delta=0.05)

x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

init_psi_guess = psi_eq_gen.generate_initial_psi_matrix()

psi_vec_func = psi_eq_gen.psi_vec_function

psi_matrix = relaxation_gauss_seidel(psi_vec_func, init_psi_guess, relaxation=1, goal_relative_error=0.01)

print(psi_matrix)

fig, ax = plt.subplots(1,1)
cp = ax.contour(x_matrix, y_matrix, psi_matrix, 50)
ax.clabel(cp, inline=True, fontsize=10)

plt.show()
