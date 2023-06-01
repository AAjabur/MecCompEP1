from utils.gauss_seidel import relaxation_gauss_seidel
from utils.mdf_equation_generator import MdfPsiEquationGenerator
from utils.sub_equations_generator import PsiSubEquationsGenerator
from utils.plot_utils import PlotUtils
import matplotlib.pyplot as plt
import numpy as np

delta = 0.02
psi_eq_gen = MdfPsiEquationGenerator(delta)

x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

psi_matrix = np.load("psi_matrix.npy")

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

fix, ax = plt.subplots()
PlotUtils.plot_quiver_decreasing_density(x_matrix, y_matrix, sub_equation_gen.x_velocity, sub_equation_gen.y_velocity, 500)
plt.show()

pressure = sub_equation_gen.rel_pressure

fig, ax = plt.subplots(1,1)
heat_map = ax.contourf(x_matrix, y_matrix, pressure, 50, cmap="hot")
fig.colorbar(heat_map)
plt.show()

along_car_border = (
    psi_eq_gen.circle_border[1:-1, 1:-1]
    |
    psi_eq_gen.circle_bottom_border[1:-1, 1:-1]
)

pressure_along_car = pressure[along_car_border]
x_values = x_matrix[along_car_border]
y_values = y_matrix[along_car_border]

print(pressure_along_car)
plt.plot(pressure_along_car)
plt.show()


