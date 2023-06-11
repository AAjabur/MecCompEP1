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
        acelerations = self.__psi_to_acceleration()
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
        left_neighbors = np.vstack((np.full(num_columns, np.nan), self.psi_matrix[:-1]))
        right_neighbors = np.vstack((self.psi_matrix[1:], np.full(num_columns, np.nan)))
        top_neighbors = np.hstack((self.psi_matrix[:,1:], np.full((num_rows, 1), np.nan)))
        bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), self.psi_matrix[:,:-1]))

        # using the second central difference
        x_velocity = (top_neighbors - bottom_neighbors)/(2*self.delta)
        y_velocity = -(right_neighbors - left_neighbors)/(2*self.delta)

        # Left and right border y velocities are zero
        y_velocity[0, :] = 0
        y_velocity[-1, :] = 0

        # Top border velocitie is V
        x_velocity[:, -1] = self.psi_equation_gen.V
    
        # Use taylor on bottom border
        x_velocity[:, 0] = top_neighbors[:, 0] / self.delta

        left_circle = (self.psi_equation_gen.circle_border & (self.psi_equation_gen.i_index_matrix*self.delta <= self.psi_equation_gen.x_size/2))
        right_circle = (self.psi_equation_gen.circle_border & (self.psi_equation_gen.i_index_matrix*self.delta > self.psi_equation_gen.x_size/2))
        circle_bottom = self.psi_equation_gen.circle_bottom_border

        # in the borders of the circle use irregular contour of the first derivative

        # For the left circle
        b = (
            self.psi_equation_gen.j_index_matrix[left_circle]*self.delta
            -
            self.psi_equation_gen.h
            -
            np.sqrt(
                (self.psi_equation_gen.L/2)**2
                -
                (self.psi_equation_gen.d + self.psi_equation_gen.L/2 - self.psi_equation_gen.i_index_matrix[left_circle]*self.delta)**2
            )
        ) / self.delta
        vertical_irregular = b < 1 # when b is NaN is not irregular
        b = b[vertical_irregular]

        g = (
            self.psi_equation_gen.d 
            + 
            self.psi_equation_gen.L/2 
            - 
            self.psi_equation_gen.i_index_matrix[left_circle]*self.delta
            -
            np.sqrt(
                (self.psi_equation_gen.L/2)**2 
                - 
                (self.psi_equation_gen.j_index_matrix[left_circle]*self.delta - self.psi_equation_gen.h)**2
            )
        ) / self.delta
        horizontal_irregular = g < 1 # when g is NaN is not irregular
        g = g[horizontal_irregular]

        y_velocity[left_circle][horizontal_irregular] = (
            g**2*left_neighbors[left_circle][horizontal_irregular]
            +
            self.psi_matrix[left_circle][horizontal_irregular]*(1-g**2)
        ) / (self.delta*g*(1 + g))

        x_velocity[left_circle][vertical_irregular] = (
            b**2*top_neighbors[left_circle][vertical_irregular]
            +
            self.psi_matrix[left_circle][vertical_irregular]*(1-b**2)
        ) / (self.delta*b*(1 + b))

        # For the right circle

        g = (
            self.psi_equation_gen.i_index_matrix[right_circle]*self.delta
            -
            self.psi_equation_gen.d 
            - 
            self.psi_equation_gen.L/2 
            -
            np.sqrt(
                (self.psi_equation_gen.L/2)**2 
                - 
                (self.psi_equation_gen.j_index_matrix[right_circle]*self.delta - self.psi_equation_gen.h)**2
            )
        ) / self.delta
        horizontal_irregular = g < 1 # when g is NaN is not irregular
        g = g[horizontal_irregular]

        b = (
            self.psi_equation_gen.j_index_matrix[right_circle]*self.delta
            -
            self.psi_equation_gen.h
            -
            np.sqrt(
                (self.psi_equation_gen.L/2)**2
                -
                (
                    self.psi_equation_gen.d 
                    + 
                    self.psi_equation_gen.L/2 
                    - 
                    self.psi_equation_gen.i_index_matrix[right_circle]*self.delta
                )**2
            )
        ) / self.delta
        vertical_irregular = b < 1 # when b is NaN is not irregular
        b = b[vertical_irregular]

        x_velocity[right_circle][vertical_irregular] = (
            b**2*top_neighbors[right_circle][vertical_irregular]
            +
            self.psi_matrix[right_circle][vertical_irregular]*(1-b**2)
        ) / (self.delta*b*(1 + b))

        y_velocity[right_circle][horizontal_irregular] = (
            -g**2*right_neighbors[right_circle][horizontal_irregular]
            -
            self.psi_matrix[right_circle][horizontal_irregular]*(1-g**2)
        ) / (self.delta*g*(1 + g))

        # on the bottom of the circle use irregular contour of the first derivative
        a = ((self.psi_equation_gen.h - self.psi_equation_gen.j_index_matrix[circle_bottom]*self.delta) / self.delta)[0]

        if a != 0:
            x_velocity[circle_bottom] = (
                -a**2*bottom_neighbors[circle_bottom]
                -
                self.psi_matrix[circle_bottom]*(1-a**2)
            ) / (a*self.delta*(1 + a))
        else:
            x_velocity[circle_bottom] = -bottom_neighbors[circle_bottom] / self.delta

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
    
    def __psi_to_acceleration(self):
        num_rows, num_columns = self.psi_matrix.shape

        # define neighbor matrices of x velocities, insert nan in the borders
        left_neighbors = np.vstack((np.full(num_columns, np.nan), self.psi_matrix[:-1]))
        right_neighbors = np.vstack((self.psi_matrix[1:], np.full(num_columns, np.nan)))
        top_neighbors = np.hstack((self.psi_matrix[:,1:], np.full((num_rows, 1), np.nan)))
        bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), self.psi_matrix[:,:-1]))

        right_top_neighbors = np.hstack((right_neighbors[:,1:], np.full((num_rows, 1), np.nan)))
        right_bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), right_neighbors[:,:-1]))
        left_top_neighbors = np.hstack((left_neighbors[:,1:], np.full((num_rows, 1), np.nan)))
        left_bottom_neighbors = np.hstack((np.full((num_rows, 1), np.nan), left_neighbors[:,:-1]))

        x_x_acel = (right_top_neighbors - left_top_neighbors - right_bottom_neighbors + left_bottom_neighbors) / (4*self.delta**2)
        y_y_acel = - x_x_acel
    
        x_y_acel = (top_neighbors - 2*self.psi_matrix + bottom_neighbors)/(self.delta**2)

        y_x_acel = -(right_neighbors - 2*self.psi_matrix + left_neighbors)/(self.delta**2)

        left_circle = (self.psi_equation_gen.circle_border & (self.psi_equation_gen.i_index_matrix*self.delta <= self.psi_equation_gen.x_size/2))
        right_circle = (self.psi_equation_gen.circle_border & (self.psi_equation_gen.i_index_matrix*self.delta > self.psi_equation_gen.x_size/2))
        circle_bottom = self.psi_equation_gen.circle_bottom_border

        # For the left circle
        b = (
            self.psi_equation_gen.j_index_matrix[left_circle]*self.delta
            -
            self.psi_equation_gen.h
            -
            np.sqrt(
                (self.psi_equation_gen.L/2)**2
                -
                (self.psi_equation_gen.d + self.psi_equation_gen.L/2 - self.psi_equation_gen.i_index_matrix[left_circle]*self.delta)**2
            )
        ) / self.delta
        vertical_irregular = b < 1 # when b is NaN is not irregular
        b = b[vertical_irregular]

        g = (
            self.psi_equation_gen.d 
            + 
            self.psi_equation_gen.L/2 
            - 
            self.psi_equation_gen.i_index_matrix[left_circle]*self.delta
            -
            np.sqrt(
                (self.psi_equation_gen.L/2)**2 
                - 
                (self.psi_equation_gen.j_index_matrix[left_circle]*self.delta - self.psi_equation_gen.h)**2
            )
        ) / self.delta
        horizontal_irregular = g < 1 # when g is NaN is not irregular
        g = g[horizontal_irregular]

        x_y_acel[left_circle][vertical_irregular] = (
            2*(b*top_neighbors[left_circle][vertical_irregular] - self.psi_matrix[left_circle][vertical_irregular]*(1+b))
        ) / (
            b*self.delta**2*(b + 1)
        )

        y_x_acel[left_circle][horizontal_irregular] = -(
            2*(g*right_neighbors[left_circle][horizontal_irregular] - self.psi_matrix[left_circle][horizontal_irregular]*(1+g))
        ) / (
            g*self.delta**2*(g + 1)
        )

        # Calculating x_x accel and y_y accel in the left circle
        R = self.psi_equation_gen.L/2
        d_x = (
            self.psi_equation_gen.d 
            + 
            R
            - 
            self.psi_equation_gen.i_index_matrix[left_circle]*self.delta
        )
        d_y = (
            self.psi_equation_gen.j_index_matrix[left_circle] * self.delta
            -
            self.psi_equation_gen.h
        )
        alpha_1 = (
            d_x - d_y - np.sqrt(2*R**2 - (d_x + d_y)**2)
        ) / 2

        alpha_2 = (
            d_x + d_y - np.sqrt(2*R**2 - (d_x - d_y)**2)
        ) / 2

        top_right_irregular = alpha_2 <= self.delta
        bottom_right_irregular = alpha_1 <= self.delta

        # just top right irregular
        jtri = top_right_irregular & (~bottom_right_irregular)
        # just bottom right irregular
        jbri = bottom_right_irregular & (~top_right_irregular)

        # top and bottom right irregular
        tbri = bottom_right_irregular & top_right_irregular

        x_x_acel[left_circle][jtri] = (
            (alpha_2[jtri]**2 + alpha_2[jtri])*(left_bottom_neighbors[left_circle][jtri])
            -
            (2) * (left_top_neighbors[left_circle][jtri] + alpha_2[jtri]*right_bottom_neighbors[left_circle][jtri])
        ) / (
            2*self.delta**2*(alpha_2[jtri]**2 + alpha_2[jtri])*(2)
        )

        x_x_acel[left_circle][jbri] = (
            (2)*(right_top_neighbors[left_circle][jbri] + alpha_1[jbri] * left_bottom_neighbors[left_circle][jbri])
            -
            (alpha_1[jbri]**2 + alpha_1[jbri]) * left_top_neighbors[left_circle][jbri]
        ) / (
            2*self.delta**2*(2)*(alpha_1[jbri]**2 + alpha_1[jbri])
        )

        x_x_acel[left_circle][tbri] = (
            (alpha_2[tbri]**2 + alpha_2[tbri])*(alpha_1[tbri] * left_bottom_neighbors[left_circle][tbri])
            -
            (alpha_1[tbri]**2 + alpha_1[tbri]) * left_top_neighbors[left_circle][tbri]
        ) / (
            2*self.delta**2*(alpha_2[tbri]**2 + alpha_2[tbri])*(alpha_1[tbri]**2 + alpha_1[tbri])
        )

        y_y_acel[left_circle] = -x_x_acel[left_circle]

        # now we take advantage of the symetry
        if num_rows%2 == 0:
            x_x_acel[int(num_rows/2):] = -np.flip(x_x_acel[:int(num_rows/2)], axis=0)
        else:
            x_x_acel[int(num_rows/2)+1:] = -np.flip(x_x_acel[:int(num_rows/2)], axis=0)

        y_y_acel = -x_x_acel

        x_x_acel[:,:] = 0
        y_y_acel[:,:] = 0
        x_y_acel[:,:] = 0
        y_x_acel[:,:] = 0

        return (x_x_acel, x_y_acel, y_x_acel, y_y_acel)
    
    def __pressure_on_circle_real_border(self):
        ################### Processing the circular part of the circle

        # we must slice the matrices because the acceleration is sliced
        circle_border = self.psi_equation_gen.circle_border
        i_index_matrix = self.psi_equation_gen.i_index_matrix
        j_index_matrix = self.psi_equation_gen.j_index_matrix

        # the coordinates of the points near the circle border
        circle_border_x_values = i_index_matrix[circle_border]*self.delta
        circle_border_y_values = j_index_matrix[circle_border]*self.delta

        # the velocities of the points near the circle border
        # again we use the slice so that the velocity matrix has the same size as the acceleration matrix
        circle_border_x_vel_values = self.x_velocity[circle_border]
        circle_border_y_vel_values = self.y_velocity[circle_border]

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
        circle_bottom_border = self.psi_equation_gen.circle_bottom_border

        i_index_matrix = self.psi_equation_gen.i_index_matrix
        j_index_matrix = self.psi_equation_gen.j_index_matrix

        circle_bottom_border_x_values = i_index_matrix[circle_bottom_border]*self.delta
        circle_bottom_border_y_values = j_index_matrix[circle_bottom_border]*self.delta

        circle_bottom_border_x_vel_values = self.x_velocity[circle_bottom_border]
        circle_bottom_border_y_vel_values = self.y_velocity[circle_bottom_border]

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
