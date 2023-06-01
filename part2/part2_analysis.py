from utils.gauss_seidel import relaxation_gauss_seidel
from utils.mdf_equation_generator import MdfPsiEquationGenerator
import matplotlib.pyplot as plt
import numpy as np

def psi_to_velocity(psi_matrix, delta):
    num_rows, num_columns = psi_matrix.shape

    left_neighbors = np.vstack((np.full(num_columns, np.nan), psi_matrix[:-1]))[1:-1,1:-1]
    right_neighbors = np.vstack((psi_matrix[1:], np.full(num_columns, np.nan)))[1:-1,1:-1]
    top_neighbors = np.hstack((psi_matrix[:,1:], np.full((num_rows, 1), np.nan)))[1:-1,1:-1]
    bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), psi_matrix[:,:-1]))[1:-1,1:-1]


    x_velocity = (top_neighbors - bottom_neighbors)/(2*delta)
    y_velocity = -(right_neighbors - left_neighbors)/(2*delta)

    return (x_velocity, y_velocity)

delta = 0.1
psi_eq_gen = MdfPsiEquationGenerator(delta)

x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

psi_matrix = np.load("psi_matrix.npy")

print(psi_matrix)

fig, ax = plt.subplots(1,1)
cp = ax.contour(x_matrix, y_matrix, psi_matrix, 20)
ax.clabel(cp, inline=True, fontsize=10)

plt.show()

x_vel, y_vel = psi_to_velocity(psi_matrix, delta)
x_matrix = x_matrix[1:-1, 1:-1]
y_matrix = y_matrix[1:-1, 1:-1]

fix, ax = plt.subplots()
ax.quiver(x_matrix, y_matrix, x_vel, y_vel, headwidth=1)

plt.show()
