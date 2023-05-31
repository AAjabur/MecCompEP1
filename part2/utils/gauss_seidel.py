import numpy as np
from typing import Callable

def relaxation_gauss_seidel(
        func_matrix: Callable[[np.array], np.array],
        initial_guess: np.array,
        relaxation: float = 1.85,
        goal_relative_error: float = 0.01,
    ):
    relative_error = 100
    approx_result = np.copy(initial_guess)
    iteration = 0

    while iteration < 50:
        iteration += 1
        last_approx_result = np.copy(approx_result)
        approx_result = relaxation * func_matrix(approx_result) + (1 - relaxation)*approx_result

        relative_error = np.amax(abs(approx_result - last_approx_result) / approx_result)

        if iteration > 10000:
            raise Exception("No convergence")
    
    print(f"Gauss seidel completed in {iteration} iterations")

    return approx_result