import numpy as np
import matplotlib.pyplot as plt

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

    def generate_initial_psi_matrix(self):
        return np.ones((self.num_rows, self.num_columns))*70

    def psi_vec_function(self, psi_matrix):

        # define mask matrices to regions with diferent equations
        left_border = self.i_index_matrix == 0
        right_border =  self.i_index_matrix == self.num_rows - 1
        top_border = self.j_index_matrix == self.num_columns - 1
        bottom_border = self.j_index_matrix == 0

        # define mask matrices to the vertices of the grid
        top_left_border = left_border & top_border
        bottom_left_border = left_border & bottom_border
        top_right_border = top_border & right_border
        bottom_right_border = bottom_border & right_border

        # remove the vertices of the border regions
        left_border = left_border & (~(top_left_border | bottom_left_border))
        right_border = right_border & (~(top_right_border | bottom_right_border))
        top_border = top_border & (~(top_left_border | top_right_border))
        bottom_border = bottom_border & (~(bottom_left_border | bottom_right_border))

        circle_bottom_border = (
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

        circle_border = (
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

        inside_circle = (distance_to_circle_center < self.L/2) & (self.j_index_matrix*self.delta > self.h)

        regular_points = ~(
            circle_border | circle_bottom_border | left_border 
            |
            right_border | top_border | bottom_border 
            | 
            inside_circle | top_left_border | top_right_border | bottom_left_border | bottom_right_border
        )

        left_neighbors = np.vstack((np.full(self.num_columns, np.nan), psi_matrix[:-1]))
        right_neighbors = np.vstack((psi_matrix[1:], np.full(self.num_columns, np.nan)))
        top_neighbors = np.hstack((psi_matrix[:,1:], np.full((self.num_rows, 1), np.nan)))
        bottom_neighbors = np.hstack((np.full((self.num_rows, 1), np.nan), psi_matrix[:,:-1]))
        
        self.__proccess_left_border(left_border, psi_matrix, top_neighbors, right_neighbors, bottom_neighbors)
        self.__proccess_right_border(right_border, psi_matrix, left_neighbors, bottom_neighbors, top_neighbors)
        self.__proccess_bottom_border(bottom_border, psi_matrix)
        self.__proccess_top_border(top_border, psi_matrix, left_neighbors, right_neighbors, bottom_neighbors)
        self.__proccess_vertices(
            top_left_border, 
            top_right_border, 
            bottom_right_border, 
            bottom_left_border, 
            psi_matrix,
            left_border,
            right_border,
            top_neighbors,
            bottom_neighbors
        )
        self.__proccess_circle_bottom_border(circle_bottom_border, psi_matrix, right_neighbors, left_neighbors, bottom_neighbors)
        self.__proccess_circle_border(circle_border, psi_matrix, left_neighbors, right_neighbors, bottom_neighbors, top_neighbors)
        self.__proccess_regular_points(regular_points, psi_matrix, left_neighbors, right_neighbors, bottom_neighbors, top_neighbors)
        self.__proccess_inside_circle(inside_circle, psi_matrix)

        return psi_matrix

    def __proccess_left_border(self, left_border, psi_matrix, top_neighbors, right_neighbors, bottom_neighbors):
        psi_matrix[left_border] =  (
            (
                top_neighbors[left_border] 
                + 
                2*right_neighbors[left_border]
                + 
                bottom_neighbors[left_border]
            ) / 4
        )

    def __proccess_right_border(self, right_border, psi_matrix, left_neighbors, bottom_neighbors, top_neighbors):
        k = ((self.x_size - self.i_index_matrix[right_border]*self.delta)/self.delta)[0]

        psi_matrix[right_border] = (
            left_neighbors[right_border]/(k + 1/2) + bottom_neighbors[right_border] + top_neighbors[right_border]
        ) / (1/(k+1/2) + 2)

    def __proccess_top_border(self, top_border, psi_matrix, left_neighbors, right_neighbors, bottom_neighbors):
        k = ((self.y_size - self.j_index_matrix[top_border]*self.delta)/self.delta)[0]

        psi_matrix[top_border] = (
            left_neighbors[top_border] + right_neighbors[top_border] + (bottom_neighbors[top_border] + self.V*self.delta)/(k + 1/2)
        ) / (2 + 1 / (k+1/2))

    def __proccess_bottom_border(self, bottom_border, psi_matrix):
        psi_matrix[bottom_border] = 0

    def __proccess_vertices(
            self,
            top_left_border, 
            top_right_border, 
            bottom_right_border, 
            bottom_left_border, 
            psi_matrix,
            left_border,
            right_border,
            top_neighbors,
            bottom_neighbors
            ):
        psi_matrix[bottom_left_border] = 0
        psi_matrix[bottom_right_border] = 0
        psi_matrix[top_left_border] = 0
        psi_matrix[top_right_border] = 0

    def __proccess_circle_bottom_border(self, circle_bottom_border, psi_matrix, right_neighbors, left_neighbors, bottom_neighbors):
        a = ((self.h - self.j_index_matrix*self.delta) / self.delta)[circle_bottom_border]
        psi_matrix[circle_bottom_border] = (
            (
                right_neighbors[circle_bottom_border] 
                + 
                left_neighbors[circle_bottom_border] 
                + 
                (2*bottom_neighbors[circle_bottom_border]) / (a+1)
            ) 
            / 
            (2/a + 2)
        )

    def __proccess_circle_border(self, circle_border, psi_matrix, left_neighbors, right_neighbors, bottom_neighbors, top_neighbors):
        ################################## LEFT CIRCLE BORDER
    
        left_circle_border = (
            circle_border 
            & 
            (
                self.i_index_matrix*self.delta < self.d + self.L/2
            )
        )
        g = (
            self.d 
            + 
            self.L/2 
            - 
            self.i_index_matrix[left_circle_border]*self.delta
            -
            np.sqrt(
                (self.L/2)**2 
                - 
                (self.j_index_matrix[left_circle_border]*self.delta - self.h)**2
            )
        ) / self.delta
        horizontal_irregular = g < 1 # when g is NaN is not irregular

        b = (
            self.j_index_matrix[left_circle_border]*self.delta
            -
            self.h
            -
            np.sqrt(
                (self.L/2)**2
                -
                (self.d + self.L/2 - self.i_index_matrix[left_circle_border]*self.delta)**2
            )
        ) / self.delta
        vertical_irregular = b < 1 # when b is NaN is not irregular

        h_v_irregular = horizontal_irregular & vertical_irregular # irregular horizontaly and verticaly
        h_irregular = horizontal_irregular & (~vertical_irregular) # irregular just horizontaly
        v_irregular = vertical_irregular & (~horizontal_irregular) # irregular just verticaly

        psi_matrix[left_circle_border][h_v_irregular] = (
            2*g[h_v_irregular]*left_neighbors[left_circle_border][h_v_irregular] 
            /
            ( g[h_v_irregular]*(1+g[h_v_irregular]) )
            +
            2*(b[h_v_irregular]*top_neighbors[left_circle_border][h_v_irregular])
            /
            (b[h_v_irregular]*(1+b[h_v_irregular]))            
        ) / (2/g[h_v_irregular] + 2/b[h_v_irregular])

        psi_matrix[left_circle_border][h_irregular] = (
            2*g[h_irregular]*left_neighbors[left_circle_border][h_irregular] 
            /
            ( g[h_irregular]*(1+g[h_irregular]) )
            +
            (bottom_neighbors[left_circle_border][h_irregular] +  top_neighbors[left_circle_border][h_irregular])
        ) / (2/g[h_irregular] + 2)

        psi_matrix[left_circle_border][v_irregular] = (
            (right_neighbors[left_circle_border][v_irregular] + left_neighbors[left_circle_border][v_irregular])
            +
            2*(b[v_irregular]*top_neighbors[left_circle_border][v_irregular])
            /
            (b[v_irregular]*(1+b[v_irregular]))
        ) / (2 + 2/b[v_irregular])
    
        ################################## RIGHT CIRCLE BORDER

        right_circle_border = (
            circle_border
            &
            (
                self.i_index_matrix*self.delta >= self.d + self.L/2
            )
        )

        g = (
            self.i_index_matrix[right_circle_border]*self.delta
            -
            self.d 
            - 
            self.L/2 
            -
            np.sqrt(
                (self.L/2)**2 
                - 
                (self.j_index_matrix[right_circle_border]*self.delta - self.h)**2
            )
        ) / self.delta
        horizontal_irregular = g < 1 # when g is NaN is not irregular

        b = (
            self.j_index_matrix[right_circle_border]*self.delta
            -
            self.h
            -
            np.sqrt(
                (self.L/2)**2
                -
                (self.d + self.L/2 - self.i_index_matrix[right_circle_border]*self.delta)**2
            )
        ) / self.delta
        vertical_irregular = b < 1 # when b is NaN is not irregular

        h_v_irregular = horizontal_irregular & vertical_irregular # irregular horizontaly and verticaly
        h_irregular = horizontal_irregular & (~vertical_irregular) # irregular just horizontaly
        v_irregular = vertical_irregular & (~horizontal_irregular) # irregular just verticaly

        psi_matrix[right_circle_border][h_v_irregular] = (
            2*g[h_v_irregular]*right_neighbors[right_circle_border][h_v_irregular] 
            /
            ( g[h_v_irregular]*(1+g[h_v_irregular]) )
            +
            2*(b[h_v_irregular]*top_neighbors[right_circle_border][h_v_irregular])
            /
            (b[h_v_irregular]*(1+b[h_v_irregular]))            
        ) / (2/g[h_v_irregular] + 2/b[h_v_irregular])

        psi_matrix[right_circle_border][h_irregular] = (
            2*g[h_irregular]*right_neighbors[right_circle_border][h_irregular] 
            /
            ( g[h_irregular]*(1+g[h_irregular]) )
            +
            (bottom_neighbors[right_circle_border][h_irregular] +  top_neighbors[right_circle_border][h_irregular])
        ) / (2/g[h_irregular] + 2)

        psi_matrix[right_circle_border][v_irregular] = (
            (left_neighbors[right_circle_border][v_irregular] + right_neighbors[right_circle_border][v_irregular])
            +
            2*(b[v_irregular]*top_neighbors[right_circle_border][v_irregular])
            /
            (b[v_irregular]*(1+b[v_irregular]))
        ) / (2 + 2/b[v_irregular])

    def __proccess_regular_points(self, regular_points, psi_matrix, left_neighbors, right_neighbors, bottom_neighbors, top_neighbors):
        psi_matrix[regular_points] = (
            right_neighbors[regular_points] 
            + 
            left_neighbors[regular_points] 
            + 
            top_neighbors[regular_points] 
            + 
            bottom_neighbors[regular_points]
        ) / 4

    def __proccess_inside_circle(self, inside_circle, psi_matrix):
        psi_matrix[inside_circle] = 0


