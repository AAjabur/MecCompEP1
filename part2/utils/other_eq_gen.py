from .sub_equations_generator import PsiSubEquationsGenerator
import numpy as np
class MdfTempEquationGenerator:
    def __init__(
            self,
            psi_sub_eq_gen: PsiSubEquationsGenerator,
            T_motor = 80 + 273.15,    # K
            T_inside = 25 + 273.15,     # K
            T_outside = 20 + 273.15,    # K
            k = 0.026,    # SI
            rho = 1.25,   # SI
            c_p = 1002,   # SI
    ):
        self.delta = psi_sub_eq_gen.psi_equation_gen.delta

        self.x_velocities = psi_sub_eq_gen.x_velocity
        self.y_velocities = psi_sub_eq_gen.y_velocity

        # u_s is -1 if u>0 and 1 if u<0
        self.u_s = (self.x_velocities>0)*(-1) + (self.x_velocities<=0)*1
        self.v_s = (self.y_velocities>0)*(-1) + (self.y_velocities<=0)*1

        self.T_motor = T_motor
        self.T_inside = T_inside
        self.T_outside = T_outside
        self.k = k
        self.rho = rho
        self.c_p = c_p

        psi_eq_gen = psi_sub_eq_gen.psi_equation_gen
        self.h = psi_eq_gen.h
        self.H = psi_eq_gen.H
        self.d = psi_eq_gen.d
        self.L = psi_eq_gen.L
        self.x_size = self.L + 2*self.d

        self.i_index_matrix = psi_eq_gen.i_index_matrix
        self.j_index_matrix = psi_eq_gen.j_index_matrix

        self.left_border = psi_eq_gen.left_border
        self.right_border = psi_eq_gen.right_border
        self.top_border = psi_eq_gen.top_border
        self.bottom_border = psi_eq_gen.bottom_border

        self.bottom_left_border = psi_eq_gen.bottom_left_border
        self.bottom_right_border = psi_eq_gen.bottom_right_border
        self.top_left_border = psi_eq_gen.top_left_border
        self.top_right_border = psi_eq_gen.top_right_border

        self.inside_circle = psi_eq_gen.inside_circle
        self.circle_bottom_border = psi_eq_gen.circle_bottom_border
        self.circle_border = psi_eq_gen.circle_border

        self.regular_points = psi_eq_gen.regular_points

    def generate_initial_guess(self):
        return np.ones_like(self.x_velocities)*273.15

    def iterate_temp(self, i, j, temp_matrix):
        # depending on the position of the point, use different processing
        if self.bottom_border[i, j]:
            self.__process_bottom_border(temp_matrix, i, j)
            if (i==10 and j==4):
                print("EMBAIXO")
        elif self.top_border[i, j]:
            self.__proccess_top_border(temp_matrix, i, j)
            if (i==10 and j==4):
                print("EM CIMA")
        elif self.left_border[i, j]:
            self.__proccess_left_border(temp_matrix, i, j)
            if (i==10 and j==4):
                print("ESQUERDA")
        elif self.right_border[i, j]:
            self.__proccess_right_border(temp_matrix, i, j)
            if (i==10 and j==4):
                print("DIREITA")
        elif self.bottom_right_border[i,j]:
            temp_matrix[i,j] = self.T_outside
            if (i==10 and j==4):
                print("CU")
        elif self.bottom_left_border[i,j]:
            temp_matrix[i, j] = self.T_outside
            if (i==10 and j==4):
                print("CU")
        elif self.top_left_border[i,j]:
            temp_matrix[i,j] = self.T_outside
            if (i==10 and j==4):
                print("CU")
        elif self.top_right_border[i,j]:
            temp_matrix[i,j] = self.T_outside
            if (i==10 and j==4):
                print("CU")
        elif self.inside_circle[i, j]:
            self.__process_inside_circle(temp_matrix, i, j)
            if (i==10 and j==4):
                print("TO DENTRO")
        elif self.circle_border[i, j]:
            self.__proccess_circle_border(temp_matrix, i, j)
        elif self.circle_bottom_border[i,j]:
            self.__proccess_circle_bottom_border(temp_matrix, i, j)
            if (i==10 and j==4):
                print("TO EMBAIXO")
        elif self.regular_points[i, j]:
            self.__proccess_regular_points(temp_matrix, i, j)
            if (i==10 and j==4):
                print("SOU REGULAR")
        else:
            raise Exception()

    def __process_inside_circle(self, temp_matrix, i, j):
        angle = np.arctan2(
            self.j_index_matrix[i,j]*self.delta - self.h,
            self.i_index_matrix[i,j]*self.delta - self.d - self.L/2
        )

        if angle < np.pi/3:
            temp_matrix[i,j] = self.T_motor
        else:
            temp_matrix[i,j] = self.T_inside

    def __process_bottom_border(self, temp_matrix, i, j):
        temp_matrix[i, j] = (
            self.k
            *
            (
                temp_matrix[i+1, j] + 2*temp_matrix[i, j+1] + temp_matrix[i-1, j]
            ) / self.delta
            -
            self.rho * self.c_p * self.x_velocities[i, j] * self.u_s[i, j]
            *
            temp_matrix[i + self.u_s[i, j], j]
        ) / (
            4*self.k/self.delta - self.rho*self.c_p*self.x_velocities[i,j]*self.u_s[i,j]
        )

    def __proccess_left_border(self, temp_matrix, i, j):
        temp_matrix[i, j] =  self.T_outside

    def __proccess_right_border(self, temp_matrix, i, j):
        temp_matrix[i, j] = (
            self.k 
            *
            (
                2*temp_matrix[i-1,j] + temp_matrix[i, j-1] + temp_matrix[i, j+1]    
            ) / self.delta
            -
            self.rho * self.c_p * self.y_velocities[i, j] * self.v_s[i, j]
            *
            temp_matrix[i, j+self.v_s[i,j]]
        ) / (
            4*self.k / self.delta
            -
            self.rho*self.c_p*self.y_velocities[i,j]*self.v_s[i,j]
        )

    def __proccess_top_border(self, temp_matrix, i, j):
        temp_matrix[i, j] = (
            self.k / self.delta
            *
            (
                temp_matrix[i-1, j]
                +
                temp_matrix[i+1, j]
                +
                2*temp_matrix[i, j-1]
            )
            -
            self.rho 
            * 
            self.c_p 
            * 
            self.x_velocities[i, j] 
            * 
            self.u_s[i, j]
            *
            temp_matrix[i + self.u_s[i, j], j]
        ) / (
            4*(self.k / self.delta) 
            -
            self.rho * self.c_p * self.x_velocities[i,j] * self.u_s[i,j]
        )

    def __proccess_circle_bottom_border(self, temp_matrix, i, j):
        a = ((self.h - self.j_index_matrix[i, j]*self.delta) / self.delta)
        border_temp = (
            (i*self.delta > self.d + self.L/2)*self.T_motor
            +
            (i*self.delta <= self.d + self.L/2)*self.T_inside
        )
        temp_matrix[i, j] = (
            self.k
            *
            (
                (temp_matrix[i+1,j] + temp_matrix[i-1,j])/self.delta
                +
                2*(
                    border_temp + a*temp_matrix[i,j-1]
                ) / (self.delta*a*(1+a))
            )
            -
            self.rho * self.c_p
            *
            (
                self.x_velocities[i,j] * self.u_s[i,j] 
                *
                temp_matrix[i+self.u_s[i,j],j]
                +
                border_temp
            )
        ) / (
            2*self.k / self.delta
            +
            2*self.k / (self.delta * a)
            -
            self.rho * self.c_p * self.x_velocities[i,j] * self.u_s[i,j]
            -
            self.rho * self.c_p
        )

    def __proccess_circle_border(self, temp_matrix, i, j):
        if self.i_index_matrix[i,j]*self.delta < self.x_size/2:
            self.__process_left_circle_border(temp_matrix, i, j)
        else:
            self.__process_right_circle_border(temp_matrix, i, j)

    def __process_right_circle_border(self, temp_matrix, i, j):
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

        if (self.j_index_matrix[i,j]*self.delta - self.delta * b) > (self.h + self.L/2 * np.sin(np.pi/3)):
            y_border_temp = self.T_inside
        else:
            y_border_temp = self.T_motor

        if (self.i_index_matrix[i,j]*self.delta - self.delta * g) > (self.d + self.L/2 + self.L/2 * np.cos(np.pi/3)):
            x_border_temp = self.T_motor
        else:
            x_border_temp = self.T_inside

        h_v_irregular = horizontal_irregular & vertical_irregular # irregular horizontaly and verticaly
        h_irregular = horizontal_irregular & (~vertical_irregular) # irregular just horizontaly
        v_irregular = vertical_irregular & (~horizontal_irregular) # irregular just verticaly

        if h_v_irregular:
            x_border_temp = y_border_temp
            temp_matrix[i, j] = (
                self.k * 2
                *
                (
                    (x_border_temp + g*temp_matrix[i+1,j]) / (g*self.delta*(1 + g))
                    +
                    (y_border_temp + b*temp_matrix[i,j+1]) / (b*self.delta*(1 + b))
                )
                -
                self.rho * self.c_p
                *
                (
                    -x_border_temp * self.x_velocities[i,j] / g
                    -
                    y_border_temp * self.y_velocities[i,j] / b
                )
            ) / (
                2 * self.k / (g * self.delta)
                +
                2 * self.k / (b * self.delta)
                +
                self.rho * self.c_p * self.x_velocities[i,j] / g
                +
                self.rho * self.c_p * self.y_velocities[i,j] / b
            )
            print("HV NA DIREITA")
            print(f"\t g={g}, b={b}, u={self.x_velocities[i,j]}, v={self.y_velocities[i,j]}")
            print(f"\t rho times c_p = {self.rho * self.c_p}")
            print(f"\t x_border_temp = {x_border_temp}, y_border_temp = {y_border_temp}")
            print(f"\t parcela de x {x_border_temp * self.x_velocities[i,j] / g}")
            print(f"\t parcela de y {y_border_temp * self.y_velocities[i,j] / b}")
            parcela_2 = (self.rho * self.c_p
                *
                (
                    -x_border_temp * self.x_velocities[i,j] / g
                    -
                    y_border_temp * self.y_velocities[i,j] / b
                ))
            parcela_1 = (self.k * 2
                *
                (
                    (x_border_temp + g*temp_matrix[i+1,j]) / (g*self.delta*(1 + g))
                    +
                    (y_border_temp + b*temp_matrix[i,j+1]) / (b*self.delta*(1 + b))
                ))
            denominador = (
                2 * self.k / (g * self.delta)
                +
                2 * self.k / (b * self.delta)
                +
                self.rho * self.c_p * self.x_velocities[i,j] / g
                +
                self.rho * self.c_p * self.y_velocities[i,j] / b
            )
            print(f"\t parcela1 = {parcela_1}, parcela2 = {parcela_2}, denominador = {denominador}")
        
        elif h_irregular:
            if (i==10 and j==4):
                print("H NA DIREITA")
            temp_matrix[i, j] = (
                self.k * 2
                *
                (
                    (x_border_temp + g*temp_matrix[i+1,j]) / (g*self.delta*(1 + g))
                    +
                    (temp_matrix[i,j-1] + temp_matrix[i,j+1]) / (self.delta*(2))
                )
                -
                self.rho * self.c_p
                *
                (
                    -x_border_temp * self.x_velocities[i,j] / g
                    +
                    self.y_velocities[i,j] * self.v_s[i,j] * temp_matrix[i,j+self.v_s[i,j]]
                )
            ) / (
                2 * self.k / (g * self.delta)
                +
                2 * self.k / (self.delta)
                +
                self.rho * self.c_p * self.x_velocities[i,j] / g
                -
                self.rho * self.c_p * self.y_velocities[i,j] * self.v_s[i,j]
            )

        elif v_irregular:
            if (i==10 and j==4):
                print("V NA DIREITA")
            temp_matrix[i, j] = (
                self.k * 2
                *
                (
                    (temp_matrix[i+1,j] + temp_matrix[i-1,j]) / (self.delta*(2))
                    +
                    (y_border_temp + b*temp_matrix[i,j+1]) / (b*self.delta*(1 + b))
                )
                -
                self.rho * self.c_p
                *
                (
                    temp_matrix[i+self.u_s[i,j], j] * self.x_velocities[i,j] * self.u_s[i,j]
                    -
                    y_border_temp * self.y_velocities[i,j] / b
                )
            ) / (
                2 * self.k / (self.delta)
                +
                2 * self.k / (b * self.delta)
                -
                self.rho * self.c_p * self.x_velocities[i,j] * self.u_s[i,j]
                +
                self.rho * self.c_p * self.y_velocities[i,j] / b
            )

        else: # not irregular, can process regularly
            if (i==10 and j==4):
                print("REGULAR NA DIREITA")
            self.__proccess_regular_points(temp_matrix, i, j)

    def __process_left_circle_border(self, temp_matrix, i, j):
        x_border_temp = self.T_inside
        y_border_temp = self.T_inside

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
            print("HV NA ESQUERDA")
            temp_matrix[i, j] = (
                self.k * 2
                *
                (
                    (x_border_temp + g*temp_matrix[i-1,j]) / (g*self.delta*(1 + g))
                    +
                    (y_border_temp + b*temp_matrix[i,j+1]) / (b*self.delta*(1 + b))
                )
                -
                self.rho * self.c_p
                *
                (
                    x_border_temp * self.x_velocities[i,j] / g
                    -
                    y_border_temp * self.y_velocities[i,j] / b
                )
            ) / (
                2 * self.k / (g * self.delta)
                +
                2 * self.k / (b * self.delta)
                -
                self.rho * self.c_p * self.x_velocities[i,j] / g
                +
                self.rho * self.c_p * self.y_velocities[i,j] / b
            )
            parcela_2 = (self.rho * self.c_p
                *
                (
                    x_border_temp * self.x_velocities[i,j] / g
                    -
                    y_border_temp * self.y_velocities[i,j] / b
                ))
            parcela_1 = (self.k * 2
                *
                (
                    (x_border_temp + g*temp_matrix[i-1,j]) / (g*self.delta*(1 + g))
                    +
                    (y_border_temp + b*temp_matrix[i,j+1]) / (b*self.delta*(1 + b))
                ))
            denominador = (
                2 * self.k / (g * self.delta)
                +
                2 * self.k / (b * self.delta)
                -
                self.rho * self.c_p * self.x_velocities[i,j] / g
                +
                self.rho * self.c_p * self.y_velocities[i,j] / b
            )
            print(f"\t g={g}, b={b}, u={self.x_velocities[i,j]}, v={self.y_velocities[i,j]}")
            print(f"\t rho times c_p = {self.rho * self.c_p}")
            print(f"\t x_border_temp = {x_border_temp}, y_border_temp = {y_border_temp}")
            print(f"\t parcela de x {x_border_temp * self.x_velocities[i,j] / g}")
            print(f"\t parcela de y {y_border_temp * self.y_velocities[i,j] / b}")
            print(f"\t parcela1 = {parcela_1}, parcela2 = {parcela_2}, denominador = {denominador}")
        
        elif h_irregular:
            if (i==10 and j==4):
                print("H NA ESQUERDA")
            temp_matrix[i, j] = (
                self.k * 2
                *
                (
                    (x_border_temp + g*temp_matrix[i-1,j]) / (g*self.delta*(1 + g))
                    +
                    (temp_matrix[i,j-1] + temp_matrix[i,j+1]) / (self.delta*(2))
                )
                -
                self.rho * self.c_p
                *
                (
                    x_border_temp * self.x_velocities[i,j] / g
                    +
                    self.y_velocities[i,j] * self.v_s[i,j] * temp_matrix[i,j+self.v_s[i,j]]
                )
            ) / (
                2 * self.k / (g * self.delta)
                +
                2 * self.k / (self.delta)
                -
                self.rho * self.c_p * self.x_velocities[i,j] / g
                -
                self.rho * self.c_p * self.y_velocities[i,j] * self.v_s[i,j]
            )

        elif v_irregular:
            if (i==10 and j==4):
                print("V NA ESQUERDA")
            temp_matrix[i, j] = (
                self.k * 2
                *
                (
                    (temp_matrix[i+1,j] + temp_matrix[i-1,j])/ (self.delta*(2))
                    +
                    (y_border_temp + b*temp_matrix[i,j+1]) / (b*self.delta*(1 + b))
                )
                -
                self.rho * self.c_p
                *
                (
                    temp_matrix[i+self.u_s[i,j], j] * self.x_velocities[i,j] * self.u_s[i,j]
                    -
                    y_border_temp * self.y_velocities[i,j] / b
                )
            ) / (
                2 * self.k / (self.delta)
                +
                2 * self.k / (b * self.delta)
                -
                self.rho * self.c_p * self.x_velocities[i,j] * self.u_s[i,j]
                +
                self.rho * self.c_p * self.y_velocities[i,j] / b
            )

        else: # not irregular, can process regularly
            if (i==10 and j==4):
                print("REGULAR NA ESQUERDA")
            self.__proccess_regular_points(temp_matrix, i, j)

    def __proccess_regular_points(self, temp_matrix, i, j):
        temp_matrix[i, j] = (
            self.k
            *
            (
                temp_matrix[i+1,j] + temp_matrix[i-1,j]
                +
                temp_matrix[i,j+1] + temp_matrix[i,j-1]
            ) / self.delta
            -
            self.rho * self.c_p
            *
            (
                self.x_velocities[i,j] * self.u_s[i,j]
                *
                temp_matrix[i+self.u_s[i,j], j]
                +
                self.y_velocities[i,j] * self.v_s[i,j]
                *
                temp_matrix[i,j+self.v_s[i,j]]
            )
        ) / (
            4*self.k/self.delta
            -
            self.rho * self.c_p 
            *
            (
                self.x_velocities[i,j]*self.u_s[i,j]
                +
                self.y_velocities[i,j]*self.v_s[i,j]
            ) 
        )
