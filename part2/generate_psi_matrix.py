from utils.gauss_seidel import relaxation_gauss_seidel
from utils.mdf_equation_generator import MdfPsiEquationGenerator
import matplotlib.pyplot as plt
import numpy as np

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

for file_name, matrix_params in psi_matrices_options_parameters.items():
    psi_eq_gen = MdfPsiEquationGenerator(**matrix_params)

    x_matrix = psi_eq_gen.i_index_matrix*psi_eq_gen.delta
    y_matrix = psi_eq_gen.j_index_matrix*psi_eq_gen.delta

    init_psi_guess = psi_eq_gen.generate_initial_psi_matrix()

    psi_matrix = relaxation_gauss_seidel(psi_eq_gen, init_psi_guess, relaxation=1.85)

    np.save(f"test/{file_name}", psi_matrix)