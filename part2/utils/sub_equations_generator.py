import numpy as np
from .mdf_equation_generator import MdfPsiEquationGenerator
from matplotlib import pyplot as plt

class PsiSubEquationsGenerator:
    def __init__(
        self,
        psi_matrix,
        psi_equation_gen: MdfPsiEquationGenerator,
        gamma_ar = 1.4,
        rho = 1.25,
        fusca_width = 1.5
    ):
        self.psi_matrix = psi_matrix
        self.psi_equation_gen = psi_equation_gen
        self.delta = psi_equation_gen.delta
        self.V = psi_equation_gen.V
        self.gamma_ar = gamma_ar
        self.rho = rho
        self.fusca_width = fusca_width

        self.x_velocity, self.y_velocity, self.velocity_module = self.__psi_to_velocity()
        self.abs_pressure, self.rel_pressure = self.__velocity_to_pressure()

        # the accelerations will be used to estimate the pressures in the circle border
        acelerations = self.__velocity_to_aceleration()
        self.x_x_acel = acelerations[0]
        self.x_y_acel = acelerations[1]
        self.y_x_acel = acelerations[2]
        self.y_y_acel = acelerations[3]

        self.__pressure_on_circle_real_border()
        self.__proccess_forces_on_circle()
        self.__proccess_forces_on_circle_bottom_border()

    def __psi_to_velocity(self):
        num_rows, num_columns = self.psi_matrix.shape

        # define matrices with the neighbors of each element, insert nan in the borders
        left_neighbors = np.vstack((np.full(num_columns, np.nan), self.psi_matrix[:-1]))[1:-1,1:-1]
        right_neighbors = np.vstack((self.psi_matrix[1:], np.full(num_columns, np.nan)))[1:-1,1:-1]
        top_neighbors = np.hstack((self.psi_matrix[:,1:], np.full((num_rows, 1), np.nan)))[1:-1,1:-1]
        bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), self.psi_matrix[:,:-1]))[1:-1,1:-1]

        # define matrices with the second neighbors of each element, insert nan in the borders
        left_left_neighbors = np.vstack((np.full((2, num_columns), np.nan), self.psi_matrix[:-2]))[1:-1,1:-1]
        right_right_neighbors = np.vstack((self.psi_matrix[2:], np.full((2, num_columns), np.nan)))[1:-1,1:-1]
        top_top_neighbors = np.hstack((self.psi_matrix[:,2:], np.full((num_rows, 2), np.nan)))[1:-1,1:-1]
        bottom_bottom_neighbors = np.hstack((np.full((num_rows, 2), np.nan), self.psi_matrix[:,:-2]))[1:-1,1:-1]

        # using the second central difference
        x_velocity = (top_neighbors - bottom_neighbors)/(2*self.delta)
        y_velocity = -(right_neighbors - left_neighbors)/(2*self.delta)

        left_circle = (self.psi_equation_gen.circle_border & (self.psi_equation_gen.i_index_matrix*self.delta <= self.psi_equation_gen.x_size/2))[1:-1,1:-1]
        right_circle = (self.psi_equation_gen.circle_border & (self.psi_equation_gen.i_index_matrix*self.delta > self.psi_equation_gen.x_size/2))[1:-1,1:-1]
        circle_bottom = self.psi_equation_gen.circle_bottom_border[1:-1,1:-1]

        # in the borders of the circle use the regressive difference because inside the circle is out of our domain
        y_velocity[left_circle] = -(3*self.psi_matrix[1:-1,1:-1][left_circle] - 4*left_neighbors[left_circle] + left_left_neighbors[left_circle])/(2*self.delta)
        x_velocity[left_circle] = (-3*self.psi_matrix[1:-1,1:-1][left_circle] + 4*top_neighbors[left_circle] - top_top_neighbors[left_circle])/(2*self.delta)

        y_velocity[right_circle] = -(-3*self.psi_matrix[1:-1,1:-1][right_circle] + 4*right_neighbors[right_circle] - right_right_neighbors[right_circle])/(2*self.delta)
        x_velocity[right_circle] = (-3*self.psi_matrix[1:-1,1:-1][right_circle] + 4*top_neighbors[right_circle] - top_top_neighbors[right_circle])/(2*self.delta)

        x_velocity[circle_bottom] = (3*self.psi_matrix[1:-1,1:-1][circle_bottom] - 4*bottom_neighbors[circle_bottom] + bottom_bottom_neighbors[circle_bottom])/(2*self.delta)
    
        velocity_module = np.sqrt(x_velocity**2 + y_velocity**2)

        return (x_velocity, y_velocity, velocity_module)
    
    def __velocity_to_pressure(self):
        rel_pressure = (
            self.rho 
            * 
            ((self.gamma_ar - 1) / self.gamma_ar)
            *
            (
                self.V**2 / 2
                -
                np.sqrt(self.x_velocity**2 + self.y_velocity**2)**2 / 2
            )
        )
        abs_pressure = rel_pressure + 101325
        return (abs_pressure, rel_pressure)
    
    def __velocity_to_aceleration(self):
        num_rows, num_columns = self.x_velocity.shape

        left_neighbors = np.vstack((np.full(num_columns, np.nan), self.x_velocity[:-1]))[1:-1,1:-1]
        right_neighbors = np.vstack((self.x_velocity[1:], np.full(num_columns, np.nan)))[1:-1,1:-1]
        top_neighbors = np.hstack((self.x_velocity[:,1:], np.full((num_rows, 1), np.nan)))[1:-1,1:-1]
        bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), self.x_velocity[:,:-1]))[1:-1,1:-1]

        # define matrices with the second neighbors of each element, insert nan in the borders
        left_left_neighbors = np.vstack((np.full((2, num_columns), np.nan), self.x_velocity[:-2]))[1:-1,1:-1]
        right_right_neighbors = np.vstack((self.x_velocity[2:], np.full((2, num_columns), np.nan)))[1:-1,1:-1]
        top_top_neighbors = np.hstack((self.x_velocity[:,2:], np.full((num_rows, 2), np.nan)))[1:-1,1:-1]
        bottom_bottom_neighbors = np.hstack((np.full((num_rows, 2), np.nan), self.x_velocity[:,:-2]))[1:-1,1:-1]

        x_x_acel = (right_neighbors - left_neighbors)/(2*self.delta)
        x_y_acel = (top_neighbors - bottom_neighbors)/(2*self.delta)

        left_circle = (self.psi_equation_gen.circle_border & (self.psi_equation_gen.i_index_matrix*self.delta <= self.psi_equation_gen.x_size/2))[2:-2,2:-2]
        right_circle = (self.psi_equation_gen.circle_border & (self.psi_equation_gen.i_index_matrix*self.delta > self.psi_equation_gen.x_size/2))[2:-2,2:-2]
        circle_bottom = self.psi_equation_gen.circle_bottom_border[2:-2,2:-2]

        x_x_acel[left_circle] = (3*self.x_velocity[1:-1,1:-1][left_circle] - 4*left_neighbors[left_circle] + left_left_neighbors[left_circle])/(2*self.delta)
        x_x_acel[right_circle] = (-3*self.x_velocity[1:-1,1:-1][right_circle] + 4*right_neighbors[right_circle] - right_neighbors[right_circle])/(2*self.delta)

        x_y_acel[left_circle] = (-3*self.x_velocity[1:-1,1:-1][left_circle] + 4*top_neighbors[left_circle] + top_top_neighbors[left_circle])/(2*self.delta)
        x_y_acel[right_circle] = (-3*self.x_velocity[1:-1,1:-1][right_circle] + 4*top_neighbors[right_circle] + top_top_neighbors[right_circle])/(2*self.delta)

        x_y_acel[circle_bottom] = (3*self.x_velocity[1:-1,1:-1][circle_bottom] - 4*bottom_neighbors[circle_bottom] + bottom_bottom_neighbors[circle_bottom])/(2*self.delta)

        num_rows, num_columns = self.y_velocity.shape

        left_neighbors = np.vstack((np.full(num_columns, np.nan), self.y_velocity[:-1]))[1:-1,1:-1]
        right_neighbors = np.vstack((self.y_velocity[1:], np.full(num_columns, np.nan)))[1:-1,1:-1]
        top_neighbors = np.hstack((self.y_velocity[:,1:], np.full((num_rows, 1), np.nan)))[1:-1,1:-1]
        bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), self.y_velocity[:,:-1]))[1:-1,1:-1]

        # define matrices with the second neighbors of each element, insert nan in the borders
        left_left_neighbors = np.vstack((np.full((2, num_columns), np.nan), self.y_velocity[:-2]))[1:-1,1:-1]
        right_right_neighbors = np.vstack((self.y_velocity[2:], np.full((2, num_columns), np.nan)))[1:-1,1:-1]
        top_top_neighbors = np.hstack((self.y_velocity[:,2:], np.full((num_rows, 2), np.nan)))[1:-1,1:-1]
        bottom_bottom_neighbors = np.hstack((np.full((num_rows, 2), np.nan), self.y_velocity[:,:-2]))[1:-1,1:-1]

        y_x_acel = (right_neighbors - left_neighbors)/(2*self.delta)
        y_y_acel = (top_neighbors - bottom_neighbors)/(2*self.delta)

        y_x_acel[left_circle] = (3*self.y_velocity[1:-1,1:-1][left_circle] - 4*left_neighbors[left_circle] + left_left_neighbors[left_circle])/(2*self.delta)
        y_x_acel[right_circle] = (-3*self.y_velocity[1:-1,1:-1][right_circle] + 4*right_neighbors[right_circle] - right_neighbors[right_circle])/(2*self.delta)

        y_y_acel[left_circle] = (-3*self.y_velocity[1:-1,1:-1][left_circle] + 4*top_neighbors[left_circle] + top_top_neighbors[left_circle])/(2*self.delta)
        y_y_acel[right_circle] = (-3*self.y_velocity[1:-1,1:-1][right_circle] + 4*top_neighbors[right_circle] + top_top_neighbors[right_circle])/(2*self.delta)

        y_y_acel[circle_bottom] = (3*self.y_velocity[1:-1,1:-1][circle_bottom] - 4*bottom_neighbors[circle_bottom] + bottom_bottom_neighbors[circle_bottom])/(2*self.delta)

        return (x_x_acel, x_y_acel, y_x_acel, y_y_acel)
    
    def __pressure_on_circle_real_border(self):
        ################### Processing the circular part of the circle

        # we must slice the matrices because the acceleration is sliced
        circle_border = self.psi_equation_gen.circle_border[2:-2, 2:-2]
        i_index_matrix = self.psi_equation_gen.i_index_matrix[2:-2, 2:-2]
        j_index_matrix = self.psi_equation_gen.j_index_matrix[2:-2, 2:-2]

        # the coordinates of the points near the circle border
        circle_border_x_values = i_index_matrix[circle_border]*self.delta
        circle_border_y_values = j_index_matrix[circle_border]*self.delta

        # the velocities of the points near the circle border
        # again we use the slice so that the velocity matrix has the same size as the acceleration matrix
        circle_border_x_vel_values = self.x_velocity[1:-1, 1:-1][circle_border]
        circle_border_y_vel_values = self.y_velocity[1:-1, 1:-1][circle_border]

        # the distance to the center of the circle and the angle to the center of the circle of each point
        circle_border_radius = np.sqrt((circle_border_x_values - self.psi_equation_gen.L/2 - self.psi_equation_gen.d)**2 + (circle_border_y_values - self.psi_equation_gen.h)**2)
        circle_border_angles = np.arctan2(circle_border_y_values - self.psi_equation_gen.h, circle_border_x_values - self.psi_equation_gen.d - self.psi_equation_gen.L/2)

        # The differente between the distance to the center of the circle and the circle radius
        delta_radius = (circle_border_radius - self.psi_equation_gen.L/2)

        # Now we can estimate the velocities in the circle border using the accelerations in the points near the circle border
        circle_real_border_x_vel = circle_border_x_vel_values - self.x_x_acel[circle_border] * delta_radius*np.cos(circle_border_angles) - self.x_y_acel[circle_border] * delta_radius*np.sin(circle_border_angles)
        circle_real_border_y_vel = circle_border_y_vel_values - self.y_x_acel[circle_border] * delta_radius*np.cos(circle_border_angles) - self.y_y_acel[circle_border] * delta_radius*np.sin(circle_border_angles)

        # calculating the pressures in the circle border
        self.circle_real_border_pressures = (
            self.rho 
            * 
            ((self.gamma_ar - 1) / self.gamma_ar)
            *
            (
                self.V**2 / 2
                -
                np.sqrt(circle_real_border_x_vel**2 + circle_real_border_y_vel**2)**2 / 2
            )
            +
            101325
        )

        self.circle_real_border_x_values = circle_border_x_values - delta_radius*np.cos(circle_border_angles)
        self.circle_real_border_y_values = circle_border_y_values - delta_radius*np.sin(circle_border_angles)
        self.circle_real_border_angles = circle_border_angles

        ################### Processing the bottom part of the circle

        # now we repeat the process to the bottom of the circle
        circle_bottom_border = self.psi_equation_gen.circle_bottom_border[2:-2, 2:-2]

        i_index_matrix = self.psi_equation_gen.i_index_matrix[2:-2, 2:-2]
        j_index_matrix = self.psi_equation_gen.j_index_matrix[2:-2, 2:-2]

        circle_bottom_border_x_values = i_index_matrix[circle_bottom_border]*self.delta
        circle_bottom_border_y_values = j_index_matrix[circle_bottom_border]*self.delta

        circle_bottom_border_x_vel_values = self.x_velocity[1:-1, 1:-1][circle_bottom_border]
        circle_bottom_border_y_vel_values = self.y_velocity[1:-1, 1:-1][circle_bottom_border]

        delta_y = self.psi_equation_gen.h - circle_bottom_border_y_values

        self.circle_real_bottom_border_x_values = circle_bottom_border_x_values
        self.circle_real_bottom_border_y_values = circle_bottom_border_y_values + delta_y

        circle_real_bottom_border_x_vel = circle_bottom_border_x_vel_values + self.x_y_acel[circle_bottom_border]*delta_y
        circle_real_bottom_border_y_vel = circle_bottom_border_y_vel_values + self.y_y_acel[circle_bottom_border]*delta_y

        self.circle_real_bottom_border_pressures = (
            self.rho 
            * 
            ((self.gamma_ar - 1) / self.gamma_ar)
            *
            (
                self.V**2 / 2
                -
                np.sqrt(circle_real_bottom_border_x_vel**2 + circle_real_bottom_border_y_vel**2)**2 / 2
            )
            +
            101325
        )

        self.real_circle_x_values = np.concatenate((self.circle_real_border_x_values, self.circle_real_bottom_border_x_values))
        self.real_circle_y_values = np.concatenate((self.circle_real_border_y_values, self.circle_real_bottom_border_y_values))
        self.real_circle_pressures = np.concatenate((self.circle_real_border_pressures, self.circle_real_bottom_border_pressures))

    def __proccess_forces_on_circle(self):
        sorted_angles = np.argsort(self.circle_real_border_angles)

        sorted_circle_pressures = np.copy(self.circle_real_border_pressures)[sorted_angles]
        sorted_pressure_angles = np.copy(self.circle_real_border_angles)[sorted_angles]

        mean_pressures = (sorted_circle_pressures[1:] + sorted_circle_pressures[:-1])/2
        mean_angles = (sorted_pressure_angles[1:] + sorted_pressure_angles[:-1])/2
        delta_lenghts = (sorted_pressure_angles[1:] - sorted_pressure_angles[:-1])*self.psi_equation_gen.L/2

        self.total_force_on_circle = np.sum(mean_pressures * delta_lenghts * self.fusca_width * np.sin(mean_angles))

    def __proccess_forces_on_circle_bottom_border(self):
        sorted_x_values_indexes = np.argsort(self.circle_real_bottom_border_x_values)

        sorted_pressures = self.circle_real_bottom_border_pressures[sorted_x_values_indexes]
        sorted_x_values = self.circle_real_bottom_border_x_values[sorted_x_values_indexes]

        pressures_mean_values = (sorted_pressures[1:] + sorted_pressures[:-1])/2
        delta_lengths = sorted_x_values[1:] - sorted_x_values[:-1]

        # because of the size of the grid maybe we don't consider the force on the vertices
        # so we must calculate the force on them
        left_vertice_x_value = self.psi_equation_gen.d
        right_vertice_x_value = self.psi_equation_gen.d + self.psi_equation_gen.L

        force_on_left_vertice = (sorted_x_values[0] - left_vertice_x_value)*sorted_pressures[0]*self.fusca_width
        force_on_right_vertice = (right_vertice_x_value - sorted_x_values[-1])*sorted_pressures[-1]*self.fusca_width

        self.total_forces_on_circle_bottom = np.sum(pressures_mean_values * delta_lengths * self.fusca_width)
        self.total_forces_on_circle_bottom += force_on_left_vertice + force_on_right_vertice
