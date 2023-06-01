from utils.gauss_seidel import relaxation_gauss_seidel
from utils.mdf_equation_generator import MdfPsiEquationGenerator
import matplotlib.pyplot as plt
import numpy as np

delta = 0.1
psi_eq_gen = MdfPsiEquationGenerator(delta)

x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

init_psi_guess = psi_eq_gen.generate_initial_psi_matrix()

iterate_func = psi_eq_gen.iterate_psi

psi_matrix = relaxation_gauss_seidel(iterate_func, init_psi_guess, goal_relative_error=0.01)

np.save("psi_matrix.npy", psi_matrix)