import numpy as np
from typing import Callable

def relaxation_gauss_seidel(
        func_iterator,
        initial_guess: np.array,
        relaxation: float = 1.15,
        goal_relative_error: float = 0.01,
    ):
    relative_error = 100
    approx_result = np.copy(initial_guess)
    iteration = 0

    while relative_error > goal_relative_error:
        before_iteration = np.copy(approx_result)
        for i in range(len(initial_guess)):
            for j in range(len(initial_guess[0])):
                last_approx_result = np.copy(approx_result)

                func_iterator(i, j, approx_result)
                approx_result[i,j] = relaxation*approx_result[i,j] + (1-relaxation)*last_approx_result[i,j]
    
        iteration += 1

        relative_error = np.nanmax(np.abs(before_iteration - approx_result))
        print(relative_error)

    
    print(f"Gauss seidel completed in {iteration} iterations")

    return approx_result