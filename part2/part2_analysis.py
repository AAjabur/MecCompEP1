from utils.gauss_seidel import relaxation_gauss_seidel_temp
from utils.mdf_equation_generator import MdfPsiEquationGenerator
from utils.other_eq_gen import MdfTempEquationGenerator
from utils.sub_equations_generator import PsiSubEquationsGenerator
from utils.plot_utils import PlotUtils
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os

# Dict to save the parameters of each saved psi_matrix
psi_matrices_options_parameters = {
    "delta002h005.npy": {
        "delta": 0.02,
        "h": 0.05
    },
    "delta002h02.npy": {
        "delta": 0.02,
        "h": 0.2
    },
    "delta002h0025.npy": {
        "delta": 0.02,
        "h": 0.025
    },
    "delta002V75.npy": {
        "delta": 0.02,
        "V": 75
    },
    "delta002V140.npy": {
        "delta": 0.02,
        "V": 140
    },
}

text = ""
iterations = 1

# just to help the user to select a pre processed psi matrix
matrix_option_files = os.listdir("test")
for matrix_option in matrix_option_files:
    if os.path.isfile(os.path.join("test", matrix_option)):
        text += f"{iterations}: {matrix_option} \n"
        iterations += 1

matrix_file_number_choice = int(input("Escolha um arquivo de matriz para analisar \n" + text))
matrix_file = matrix_option_files[matrix_file_number_choice-1]

print(f"Você escolheu o arquivo {matrix_file}")

# This object have all parameters of the problem to calculate psi
psi_eq_gen = MdfPsiEquationGenerator(**psi_matrices_options_parameters[matrix_file])

print(np.count_nonzero(psi_eq_gen.bottom_border))

x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

psi_matrix = np.load(f"test/{matrix_file}")

fig, ax = plt.subplots(1,1)
cp = ax.contour(x_matrix, y_matrix, psi_matrix, 20)
ax.clabel(cp, inline=True, fontsize=10)

plt.show()

sub_equation_gen = PsiSubEquationsGenerator(psi_matrix, psi_eq_gen)

fig, ax = plt.subplots(1,1)
cp = ax.contourf(x_matrix, y_matrix, sub_equation_gen.velocity_module*3.6, 50, cmap="hot")
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
scatter_plot = ax.scatter(sub_equation_gen.real_circle_x_values, sub_equation_gen.real_circle_y_values, c=sub_equation_gen.real_circle_pressures - 101325, cmap="viridis")

min_pressure_index = np.argmin(sub_equation_gen.real_circle_pressures)
min_pressure_value = sub_equation_gen.real_circle_pressures[min_pressure_index]
min_x = sub_equation_gen.real_circle_x_values[min_pressure_index]
min_y = sub_equation_gen.real_circle_y_values[min_pressure_index]
ax.annotate(f"Minimum pressure {round(min_pressure_value, 3)}", xy=(min_x, min_y), textcoords="offset points", xytext=(10,40), arrowprops={"arrowstyle":"-|>"})

plt.colorbar(scatter_plot)

plt.show()

print(f"Forças no circulo {sub_equation_gen.total_force_on_circle}")
print(f"Forças embaixo {sub_equation_gen.total_forces_on_circle_bottom}")
print(f"Forças totais {sub_equation_gen.total_forces_on_circle_bottom - sub_equation_gen.total_force_on_circle} N")

temp_eq_gen = MdfTempEquationGenerator(sub_equation_gen)

initial_guess = temp_eq_gen.generate_initial_guess()

relaxation_gauss_seidel_temp(temp_eq_gen, initial_guess, )


