import numpy as np
from .mdf_equation_generator import MdfPsiEquationGenerator

class PsiSubEquationsGenerator:
    def __init__(
        self,
        psi_matrix,
        psi_equation_gen: MdfPsiEquationGenerator,
        gamma_ar = 1.4,
        rho = 1.25
    ):
        self.psi_matrix = psi_matrix
        self.delta = psi_equation_gen.delta
        self.V = psi_equation_gen.V
        self.gamma_ar = gamma_ar
        self.rho = rho
        self.x_velocity, self.y_velocity, self.velocity_module = self.__psi_to_velocity()

        self.abs_pressure, self.rel_pressure = self.__velocity_to_pressure()

    def __psi_to_velocity(self):
        num_rows, num_columns = self.psi_matrix.shape

        left_neighbors = np.vstack((np.full(num_columns, np.nan), self.psi_matrix[:-1]))[1:-1,1:-1]
        right_neighbors = np.vstack((self.psi_matrix[1:], np.full(num_columns, np.nan)))[1:-1,1:-1]
        top_neighbors = np.hstack((self.psi_matrix[:,1:], np.full((num_rows, 1), np.nan)))[1:-1,1:-1]
        bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), self.psi_matrix[:,:-1]))[1:-1,1:-1]

        x_velocity = (top_neighbors - bottom_neighbors)/(2*self.delta)
        y_velocity = -(right_neighbors - left_neighbors)/(2*self.delta)

        velocity_module = np.sqrt(x_velocity**2 + y_velocity**2)

        return (x_velocity, y_velocity, velocity_module)
    
    def __velocity_to_pressure(self):
        rel_pressure = (
            self.rho 
            * 
            ((self.gamma_ar - 1) / self.gamma_ar)
            *
            (
                self.V**2/2
                -
                np.sqrt(self.x_velocity**2 + self.y_velocity**2) / 2
            )
        )
        abs_pressure = rel_pressure + 101325
        return (abs_pressure, rel_pressure)
