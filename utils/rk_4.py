from typing import List, Callable, Tuple
from matplotlib import pyplot as plt
import numpy as np

def rk_4_solve(
        vec_function: Callable[[float, np.array], np.array], 
        init_x_vec: np.array,
        t_0: float,
        h: float,
        t_f: float
        ):
    number_of_iterations = int((t_f - t_0) // h) + 1

    n = 0
    t_n = t_0
    x_vec_n = init_x_vec

    t_vec = np.zeros(number_of_iterations)
    x_matrix = np.zeros((len(init_x_vec), number_of_iterations))
    for i in range(number_of_iterations):
        t_vec[n] = t_n
        x_matrix[:, n] = x_vec_n

        k1 = vec_function(t_n, x_vec_n)
        k2 = vec_function(t_n + h / 2, x_vec_n + h/2 * k1)
        k3 = vec_function(t_n + h / 2, x_vec_n + h/2 * k2)
        k4 = vec_function(t_n + h, x_vec_n + h * k3)

        x_vec_n = x_vec_n + h / 6 * (k1 + 2*k2 + 2*k3 + k4)
        t_n += h
        n += 1
    
    return (t_vec, x_matrix)

def analyze_rk_4_error(
        vec_function: Callable[[float, np.array], np.array], 
        init_x_vec: np.array,
        t_0: float,
        h: float,
        t_f: float
):
    t_vec_h1, x_matrix_h1 = rk_4_solve(vec_function, init_x_vec, t_0, h, t_f)
    t_vec_h2, x_matrix_h2 = rk_4_solve(vec_function, init_x_vec, t_0, h/2, t_f)

    x_matrix_h2 = x_matrix_h2[:, ::2]

    x_matrix_best_approximation = (x_matrix_h1 - (2**4) * x_matrix_h2) / (1 - 2**4)

    error_matrix = 16/15 * (x_matrix_h2 - x_matrix_h1)

    relative_error_matrix = np.divide(error_matrix, x_matrix_best_approximation, out=np.zeros_like(error_matrix), where=x_matrix_best_approximation!=0)

    return (t_vec_h1, relative_error_matrix)

def get_y_matrix_from_x_matrix(
        vec_function: Callable[[float, np.array], np.array],
        t_vec: np.array,
        x_matrix: np.array
):
    transposed_x_matrix = x_matrix.copy().transpose()

    y_matrix = np.zeros_like(x_matrix)
    print(y_matrix)
    for n, x_vec in enumerate(transposed_x_matrix):
        y_matrix[:, n] = vec_function(t_vec[n], x_vec)

    return y_matrix




