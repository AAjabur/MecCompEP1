import numpy as np
import matplotlib.pyplot as plt
import warnings

class MdfPsiEquationGenerator:
    def __init__(
        self,
        delta,
        V = 100, # km/h
        h = 0.15, # m
        L = 3, # m
        d = 1.5, # m
        H = 6, # m
    ):
        self.delta = delta
        self.V = V / 3.6
        self.h = h
        self.L = L
        self.d = d
        self.H = H

        self.x_size = (2*self.d + self.L)
        self.y_size = self.H

        self.num_rows = int(self.x_size / self.delta) + 1
        self.num_columns = int(self.y_size / self.delta) + 1

        self.i_index_matrix = np.tile(np.arange(self.num_rows)[:, np.newaxis], (1, self.num_columns))
        self.j_index_matrix = np.tile(np.arange(self.num_columns), (self.num_rows,1))

        # define mask matrices to regions with diferent equations
        left_border = self.i_index_matrix == 0
        right_border =  self.i_index_matrix == self.num_rows - 1
        top_border = self.j_index_matrix == self.num_columns - 1
        bottom_border = self.j_index_matrix == 0

        # define mask matrices to the vertices of the grid
        self.top_left_border = left_border & top_border
        self.bottom_left_border = left_border & bottom_border
        self.top_right_border = top_border & right_border
        self.bottom_right_border = bottom_border & right_border

        # remove the vertices of the border regions
        self.left_border = left_border & (~(self.top_left_border | self.bottom_left_border))
        self.right_border = right_border & (~(self.top_right_border | self.bottom_right_border))
        self.top_border = top_border & (~(self.top_left_border | self.top_right_border))
        self.bottom_border = bottom_border & (~(self.bottom_left_border | self.bottom_right_border))

        self.circle_bottom_border = (
            (self.h - self.j_index_matrix * self.delta < self.delta)
            &
            (self.j_index_matrix * self.delta < self.h)
            &
            (self.i_index_matrix * self.delta > self.d)
            &
            (self.i_index_matrix * self.delta < self.d + self.L)
        )

        distance_to_circle_center = (
            np.sqrt(
                (self.i_index_matrix*self.delta - self.d - self.L / 2) ** 2 
                +
                (self.j_index_matrix*self.delta - self.h) ** 2
            ) 
        )

        self.circle_border = (
            (
                distance_to_circle_center
                -
                self.L/2 < self.delta
            )
            &
            (
                distance_to_circle_center > self.L/2
            )
            &
            (
                self.j_index_matrix * self.delta > self.h
            )
        )

        self.inside_circle = (distance_to_circle_center < self.L/2) & (self.j_index_matrix*self.delta > self.h)

        self.regular_points = ~(
            self.circle_border | self.circle_bottom_border | left_border 
            |
            self.right_border | self.top_border | self.bottom_border 
            | 
            self.inside_circle | self.top_left_border | self.top_right_border | self.bottom_left_border | self.bottom_right_border
        )

    def generate_initial_psi_matrix(self):
        return np.ones((self.num_rows, self.num_columns))*107.5
    
    def iterate_psi(self, i, j, psi_matrix):
        if self.circle_border[i, j]:
            self.__proccess_circle_border(psi_matrix, i, j)
        elif self.bottom_border[i, j]:
            psi_matrix[i, j] = 0
        elif self.top_border[i, j]:
            self.__proccess_top_border(psi_matrix, i, j)
        elif self.left_border[i, j]:
            self.__proccess_left_border(psi_matrix, i, j)
        elif self.right_border[i, j]:
            self.__proccess_right_border(psi_matrix, i, j)
        elif self.circle_bottom_border[i,j]:
            self.__proccess_circle_bottom_border(psi_matrix, i, j)
        elif (
            self.bottom_left_border[i,j]
            or
            self.bottom_right_border[i,j]
            ):
            psi_matrix[i, j] = 0
        elif self.top_left_border[i,j]:
            self.__process_top_left_border(psi_matrix, i, j)
        elif self.top_right_border[i,j]:
            self.__process_top_right_border(psi_matrix, i, j)
        elif self.inside_circle[i, j]:
            psi_matrix[i, j] = 0
        elif self.regular_points[i, j]:
            self.__proccess_regular_points(psi_matrix, i, j)

    def __process_top_left_border(self, psi_matrix, i, j):
        k = (self.y_size - self.j_index_matrix[i, j]*self.delta)
        psi_matrix[i, j] = (
            (
                2*psi_matrix[i+1, j] 
                + 
                (
                    psi_matrix[i, j-1] + 
                    self.delta*self.V
                ) / (k + 1/2)
            )
            /
            (2 + 1 / (k + 1/2))
        )

    def __process_top_right_border(self, psi_matrix, i, j):
        k = (self.y_size - self.j_index_matrix[i, j]*self.delta)
        psi_matrix[i,j] = (
            (
                2*psi_matrix[i-1, j] 
                + 
                (
                    psi_matrix[i, j-1] + 
                    self.delta*self.V
                ) / (k + 1/2)
            )
            /
            (2 + 1 / (k + 1/2))
        )

    def __proccess_left_border(self, psi_matrix, i, j):
        psi_matrix[i, j] =  (
            (
                psi_matrix[i, j+1] 
                + 
                2*psi_matrix[i+1, j]
                + 
                psi_matrix[i, j-1]
            ) / 4
        )

    def __proccess_right_border(self, psi_matrix, i, j):
        k = ((self.x_size - self.i_index_matrix[i, j]*self.delta)/self.delta)

        psi_matrix[i, j] = (
            psi_matrix[i-1, j]/(k + 1/2) + psi_matrix[i, j-1] + psi_matrix[i, j+1]
        ) / (1/(k+1/2) + 2)

    def __proccess_top_border(self, psi_matrix, i, j):
        k = ((self.y_size - self.j_index_matrix[i, j]*self.delta)/self.delta)

        psi_matrix[i, j] = (
            psi_matrix[i-1, j] + psi_matrix[i+1, j] + (psi_matrix[i, j-1] + self.V*self.delta)/(k + 1/2)
        ) / (2 + 1 / (k+1/2))

    def __proccess_circle_bottom_border(self, psi_matrix, i, j):
        a = ((self.h - self.j_index_matrix[i, j]*self.delta) / self.delta)
        psi_matrix[i, j] = (
            (
                psi_matrix[i+1, j] 
                + 
                psi_matrix[i-1, j] 
                + 
                (2*psi_matrix[i, j-1]) / (a+1)
            ) 
            / 
            (2/a + 2)
        )

    def __proccess_circle_border(self, psi_matrix, i, j):
        if self.i_index_matrix[i,j]*self.delta < self.x_size/2:
            self.__process_left_circle_border(psi_matrix, i, j)
        else:
            self.__process_right_circle_border(psi_matrix, i, j)

    def __process_right_circle_border(self, psi_matrix, i, j):
        g = (
            self.i_index_matrix[i,j]*self.delta
            -
            self.d 
            - 
            self.L/2 
            -
            np.sqrt(
                (self.L/2)**2 
                - 
                (self.j_index_matrix[i,j]*self.delta - self.h)**2
            )
        ) / self.delta
        horizontal_irregular = g < 1 # when g is NaN is not irregular

        b = (
            self.j_index_matrix[i,j]*self.delta
            -
            self.h
            -
            np.sqrt(
                (self.L/2)**2
                -
                (self.d + self.L/2 - self.i_index_matrix[i,j]*self.delta)**2
            )
        ) / self.delta
        vertical_irregular = b < 1 # when b is NaN is not irregular

        h_v_irregular = horizontal_irregular & vertical_irregular # irregular horizontaly and verticaly
        h_irregular = horizontal_irregular & (~vertical_irregular) # irregular just horizontaly
        v_irregular = vertical_irregular & (~horizontal_irregular) # irregular just verticaly

        if h_v_irregular:
            psi_matrix[i, j] = (
                2*g*psi_matrix[i+1, j] 
                /
                ( g*(1+g) )
                +
                2*(b*psi_matrix[i, j+1])
                /
                (b*(1+b))            
            ) / (2/g + 2/b)

        elif h_irregular:
            psi_matrix[i, j] = (
                2*g*psi_matrix[i+1, j] 
                /
                ( g*(1+g) )
                +
                (psi_matrix[i, j-1] +  psi_matrix[i, j+1])
            ) / (2/g + 2)

        elif v_irregular:
            psi_matrix[i, j] = (
                (psi_matrix[i-1, j] + psi_matrix[i+1, j])
                +
                2*(b*psi_matrix[i, j+1])
                /
                (b*(1+b))
            ) / (2 + 2/b)

        else:
            self.__proccess_regular_points(psi_matrix, i, j)

    def __process_left_circle_border(self, psi_matrix, i, j):
        g = (
            self.d 
            + 
            self.L/2 
            - 
            self.i_index_matrix[i, j]*self.delta
            -
            np.sqrt(
                (self.L/2)**2 
                - 
                (self.j_index_matrix[i, j]*self.delta - self.h)**2
            )
        ) / self.delta
        horizontal_irregular = g < 1 # when g is NaN is not irregular

        b = (
            self.j_index_matrix[i, j]*self.delta
            -
            self.h
            -
            np.sqrt(
                (self.L/2)**2
                -
                (self.d + self.L/2 - self.i_index_matrix[i, j]*self.delta)**2
            )
        ) / self.delta
        vertical_irregular = b < 1 # when b is NaN is not irregular

        h_v_irregular = horizontal_irregular & vertical_irregular # irregular horizontaly and verticaly
        h_irregular = horizontal_irregular & (~vertical_irregular) # irregular just horizontaly
        v_irregular = vertical_irregular & (~horizontal_irregular) # irregular just verticaly

        if h_v_irregular:
            psi_matrix[i, j] = (
                2*g*psi_matrix[i-1, j]
                /
                ( g*(1+g) )
                +
                2*(b*psi_matrix[i, j+1])
                /
                (b*(1+b))            
            ) / (2/g + 2/b)
        
        elif h_irregular:
            psi_matrix[i, j] = (
                2*g*psi_matrix[i-1, j] 
                /
                ( g*(1+g) )
                +
                (psi_matrix[i, j-1] +  psi_matrix[i, j+1])
            ) / (2/g + 2)

        elif v_irregular:
            psi_matrix[i,j] = (
                (psi_matrix[i+1, j] + psi_matrix[i-1, j])
                +
                2*(b*psi_matrix[i, j+1])
                /
                (b*(1+b))
            ) / (2 + 2/b)

        else:
            self.__proccess_regular_points(psi_matrix, i, j)

    def __proccess_regular_points(self, psi_matrix, i, j):
        psi_matrix[i, j] = (
            psi_matrix[i+1, j] 
            + 
            psi_matrix[i-1, j] 
            + 
            psi_matrix[i, j+1] 
            + 
            psi_matrix[i, j-1]
        ) / 4


