from utils.gauss_seidel import relaxation_gauss_seidel
from utils.mdf_equation_generator import MdfPsiEquationGenerator
from utils.sub_equations_generator import PsiSubEquationsGenerator
from utils.plot_utils import PlotUtils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os

psi_matrices_options_deltas = {
    "002step_regular.npy": 0.02,
    "h01delta004.npy": 0.04,
    "h005delta004.npy": 0.04,
    "h02delta004.npy": 0.04,
    "h0025delta005.npy": 0.05,
    "V75delta005.npy": 0.05,
    "V140delta005.npy": 0.05,
    "V140h02delta005.npy": 0.05
}

text = ""
iterations = 1

matrix_option_files = os.listdir("psi_matrices")
for matrix_option in matrix_option_files:
    if os.path.isfile(os.path.join("psi_matrices", matrix_option)):
        text += f"{iterations}: {matrix_option} \n"
        iterations += 1

matrix_file_number_choice = int(input("Escolha um arquivo de matriz para analisar \n" + text))
matrix_file = matrix_option_files[matrix_file_number_choice-1]

print(f"Você escolheu o arquivo {matrix_file}")

delta = psi_matrices_options_deltas[matrix_file]
psi_eq_gen = MdfPsiEquationGenerator(delta)

x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

psi_matrix = np.load(f"psi_matrices/{matrix_file}")

fig, ax = plt.subplots(1,1)
cp = ax.contour(x_matrix, y_matrix, psi_matrix, 20)
ax.clabel(cp, inline=True, fontsize=10)

plt.show()

sub_equation_gen = PsiSubEquationsGenerator(psi_matrix, psi_eq_gen)

x_matrix = x_matrix[1:-1, 1:-1]
y_matrix = y_matrix[1:-1, 1:-1]

fig, ax = plt.subplots(1,1)
cp = ax.contourf(x_matrix, y_matrix, sub_equation_gen.velocity_module, 50, cmap="hot")
fig.colorbar(cp)

plt.show()

PlotUtils.plot_quiver_decreasing_density(x_matrix, y_matrix, sub_equation_gen.x_velocity, sub_equation_gen.y_velocity, 500)
plt.show()

pressure = sub_equation_gen.rel_pressure

fig, ax = plt.subplots(1,1)
heat_map = ax.contourf(x_matrix, y_matrix, pressure, 50, cmap="hot")
fig.colorbar(heat_map)
plt.show()

fig, ax = plt.subplots(1, 1)
scatter_plot = ax.scatter(sub_equation_gen.real_circle_x_values, sub_equation_gen.real_circle_y_values, c=sub_equation_gen.real_circle_pressures, cmap="viridis")

min_pressure_index = np.argmin(sub_equation_gen.real_circle_pressures)
min_pressure_value = sub_equation_gen.real_circle_pressures[min_pressure_index]
min_x = sub_equation_gen.real_circle_x_values[min_pressure_index]
min_y = sub_equation_gen.real_circle_y_values[min_pressure_index]
ax.annotate(f"Minimum pressure {round(min_pressure_value, 3)}", xy=(min_x, min_y), textcoords="offset points", xytext=(10,20), arrowprops={"arrowstyle":"-|>"})

plt.colorbar(scatter_plot)

plt.show()

print(f"Forças no circulo {sub_equation_gen.total_force_on_circle}")
print(f"Forças embaixo {sub_equation_gen.total_forces_on_circle_bottom}")
print(f"Forças totais {sub_equation_gen.total_forces_on_circle_bottom + sub_equation_gen.total_force_on_circle} N")


