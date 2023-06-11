from utils.mdf_equation_generator import MdfPsiEquationGenerator
from utils.sub_equations_generator import PsiSubEquationsGenerator
from utils.other_eq_gen import MdfTempEquationGenerator
from utils.gauss_seidel import relaxation_gauss_seidel_temp, relaxation_gauss_seidel_psi
from matplotlib import pyplot as plt

psi_eq_gen = MdfPsiEquationGenerator(3/8)

init_psi_guess = psi_eq_gen.generate_initial_psi_matrix()
psi = relaxation_gauss_seidel_psi(psi_eq_gen, init_psi_guess)
sub_eq_gen = PsiSubEquationsGenerator(psi, psi_eq_gen)

x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

fig, ax = plt.subplots(1,1)
cp = ax.contourf(x_matrix, y_matrix, sub_eq_gen.velocity_module*3.6, 50, cmap="hot")
fig.colorbar(cp)
plt.show()

temp_eq_gen = MdfTempEquationGenerator(sub_eq_gen)
temp_init_guess = temp_eq_gen.generate_initial_guess()
temp = relaxation_gauss_seidel_temp(temp_eq_gen, temp_init_guess)
print(temp)


