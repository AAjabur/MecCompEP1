import numpy as np
from typing import Callable
from utils.mdf_equation_generator import MdfPsiEquationGenerator
from utils.other_eq_gen import MdfTempEquationGenerator
from matplotlib import pyplot as plt

def relaxation_gauss_seidel_psi(
        psi_eq_gen: MdfPsiEquationGenerator,
        initial_guess: np.array,
        relaxation: float = 1.15,
        goal_relative_error: float = 0.01,
    ):
    func_iterator = psi_eq_gen.iterate_psi

    relative_error = 100
    approx_result = np.copy(initial_guess)
    iteration = 0

    # Using the fact that the matrix must be symetric and the psi values inside the circle don't change
    # we can make less iterations inside the for loop
    points_to_iterate = (~psi_eq_gen.inside_circle) & (psi_eq_gen.i_index_matrix*psi_eq_gen.delta <= psi_eq_gen.x_size/2)
    
    i_values = psi_eq_gen.i_index_matrix[points_to_iterate]
    j_values = psi_eq_gen.j_index_matrix[points_to_iterate]

    num_rows = len(approx_result)
    while relative_error > goal_relative_error:
        before_iteration = np.copy(approx_result)
        for i,j in zip(i_values, j_values):
                last_approx_result = np.copy(approx_result)

                func_iterator(i, j, approx_result)
                approx_result[i,j] = relaxation*approx_result[i,j] + (1-relaxation)*last_approx_result[i,j]


        iteration += 1

        relative_error = np.nanmax(np.abs(before_iteration - approx_result))
        print(relative_error)

        if num_rows%2 == 0:
            approx_result[int(num_rows/2):] = np.flip(approx_result[:int(num_rows/2)], axis=0)
        else:
            approx_result[int(num_rows/2)+1:] = np.flip(approx_result[:int(num_rows/2)], axis=0)
    
    print(f"Gauss seidel completed in {iteration} iterations")

    return approx_result

def relaxation_gauss_seidel_temp(
    temp_eq_gen: MdfTempEquationGenerator,
    initial_guess: np.array,
    relaxation: float = 1.15,
    goal_relative_error: float = 0.01,
):
    func_iterator = temp_eq_gen.iterate_temp
    relative_error = 100
    approx_result = np.copy(initial_guess)
    iteration = 0


    x_matrix = temp_eq_gen.i_index_matrix*temp_eq_gen.delta
    y_matrix = temp_eq_gen.j_index_matrix*temp_eq_gen.delta

    num_rows, num_columns = initial_guess.shape
    while relative_error > goal_relative_error:
        before_iteration = np.copy(approx_result)
        for i in range(num_rows):
            for j in range(num_columns):
                last_approx_result = np.copy(approx_result)
                func_iterator(i, j, approx_result)

                approx_result[i,j] = relaxation*approx_result[i,j] + (1-relaxation)*last_approx_result[i,j]

                if False:
                    print(f"(i,j) = ({i},{j})")
                    fig, ax = plt.subplots(1,1)
                    cp = ax.contourf(x_matrix, y_matrix, approx_result - 273.15, 50, cmap="hot")
                    fig.colorbar(cp)
                
                    plt.show()

        iteration += 1
        relative_error = np.nanmax(np.abs(before_iteration - approx_result))
        print(relative_error)

    return approx_result

    
                    