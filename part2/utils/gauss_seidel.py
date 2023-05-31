import numpy as np
from typing import Callable

def relaxation_gauss_seidel(
        func_matrix: Callable[[np.array], np.array],
        initial_guess: np.array,
        relaxation: float = 1.15,
        goal_relative_error: float = 0.01,
    ):
    relative_error = 100
    approx_result = np.copy(initial_guess)
    iteration = 0

    while relative_error > goal_relative_error:
        iteration += 1
        last_approx_result = np.copy(approx_result)
        approx_result = relaxation * func_matrix(approx_result) + (1 - relaxation)*last_approx_result

        relative_error = np.nanmax(abs(approx_result - last_approx_result))

        if iteration % 1000 == 0:
            print(iteration)
            print(relative_error)
        if iteration > 100000000:
            print("No convergence")
            return approx_result
    
    print(f"Gauss seidel completed in {iteration} iterations")

    return approx_result