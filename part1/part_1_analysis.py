import numpy as np
from numpy import pi
from utils.rk_4 import rk_4_solve, analyze_rk_4_error, get_y_matrix_from_x_matrix
from utils.car_equation import CarVecFunctionGenerator
from utils.best_plot import find_best_plot_ten_expoent

from matplotlib import pyplot as plt

def plot_all_variables_with_step(h, vec_func, additional_title = ""):
    init_x = np.array([0, 0, 0.09, 0])

    (t_vec, x_matrix) = rk_4_solve(vec_func, init_x, 0, h, 4)
    y_matrix = get_y_matrix_from_x_matrix(vec_func, t_vec, x_matrix)
    best_10_expoents = find_best_plot_ten_expoent(x_matrix[0], x_matrix[1], y_matrix[1])

    label = r'$x.10^' + str(best_10_expoents[0]) + r'$' + r'$(m)$'
    plt.plot(t_vec, x_matrix[0] * 10 ** best_10_expoents[0], label=label, linewidth=1)
    label = r'$\dot{x}.10^' + str(best_10_expoents[1]) + r'$' + r'$(\frac{m}{s})$'
    plt.plot(t_vec, x_matrix[1] * 10 ** best_10_expoents[1], label=label, linewidth=1)
    label = r'$\ddot{x}.10^' + str(best_10_expoents[2]) + r'$' + r'$(\frac{m}{s^2})$'
    plt.plot(t_vec, y_matrix[1] * 10 ** best_10_expoents[2], label=label, linewidth=1)

    plt.legend()
    plt.xlabel("t (s)")
    plt.title(r'Valores de $x$, $\dot{x}$ e $\ddot{x}$ calculados com passo $h = ' + str(h) + r'$ ' + additional_title)

    plt.figure()

    best_10_expoents = find_best_plot_ten_expoent(x_matrix[2], x_matrix[3], y_matrix[3])
    label = r'$\theta.10^' + str(best_10_expoents[0]) + r'$' + r'$(rad)$'
    plt.plot(t_vec, x_matrix[2] * 10 ** best_10_expoents[0], label=label, linewidth=1)
    label = r'$\dot{\theta}.10^' + str(best_10_expoents[1]) + r'$' + r'$(\frac{rad}{s})$'
    plt.plot(t_vec, x_matrix[3] * 10 ** best_10_expoents[1], label=label, linewidth=1)
    label = r'$\ddot{\theta}.10^' + str(best_10_expoents[2]) + r'$' + r'$(\frac{rad}{s^2})$'
    plt.plot(t_vec, y_matrix[3] * 10 ** best_10_expoents[2], label=label, linewidth=1)

    plt.legend()
    plt.xlabel("t (s)")
    plt.title(r'Valores de $\theta$, $\dot{\theta}$ e $\ddot{\theta}$ calculados com passo $h = ' + str(h) + r'$ ' + additional_title)

def plot_just_x_with_step(h, vec_func):
    init_x = np.array([0, 0, 0.09, 0])
    (t_vec, x_matrix) = rk_4_solve(vec_func, init_x, 0, h, 4)

    label = r'$x' + r'$' + r'$(m)$'
    plt.plot(t_vec, 1000*x_matrix[0], label=label, linewidth=1)
    plt.xlabel("t (s)")
    plt.ylabel("x (mm)")

car_function_gen = CarVecFunctionGenerator()

vec_func = car_function_gen.car_vec_func

plot_all_variables_with_step(0.0001, vec_func)
plt.show()
plt.figure()
plot_all_variables_with_step(0.001, vec_func)
plt.show()
plt.figure()
plot_all_variables_with_step(0.01, vec_func)
plt.show()
plt.figure()

step = 0.0001
car_function_gen = CarVecFunctionGenerator(V=30)
car_vec_func = car_function_gen.car_vec_func
plot_all_variables_with_step(step, car_vec_func, r'$V = 30 \frac{km}{h}$')
plt.show()
plt.figure()

car_function_gen = CarVecFunctionGenerator(V=70)
car_vec_func = car_function_gen.car_vec_func
plot_all_variables_with_step(step, car_vec_func, r'$V = 70 \frac{km}{h}$')
plt.show()
plt.figure()

car_function_gen = CarVecFunctionGenerator(c1 = 10**3, c2=10**3)
car_vec_func = car_function_gen.car_vec_func
plot_all_variables_with_step(step, car_vec_func, r'$c_1 = c_2 = 10^3$')
plt.show()
plt.figure()

car_function_gen = CarVecFunctionGenerator(c1 = 5 * 10**6, c2=5 * 10**6)
car_vec_func = car_function_gen.car_vec_func
plot_all_variables_with_step(step, car_vec_func, r'$c_1 = c_2 = 5.10^6$')
plt.show()
plt.figure()

car_function_gen = CarVecFunctionGenerator(c1 = 2.5 * 10**5, c2=2.5 * 10**5)
car_vec_func = car_function_gen.car_vec_func
plot_all_variables_with_step(step, car_vec_func, r'$c_1 = c_2 = 2{,}5.10^5$')
plt.show()
plt.figure()

car_function_gen = CarVecFunctionGenerator(k1 = 2 * 10**4, k2=2 * 10**4)
car_vec_func = car_function_gen.car_vec_func
plot_all_variables_with_step(step, car_vec_func, r'$k_1 = k_2 = 2.10^4$')
plt.show()
plt.figure()

car_function_gen = CarVecFunctionGenerator(k1 = 5 * 10**6, k2=5 * 10**6)
car_vec_func = car_function_gen.car_vec_func
plot_all_variables_with_step(step, car_vec_func, r'$k_1 = k_2 = 5.10^6$')
plt.show()
plt.figure()

car_function_gen = CarVecFunctionGenerator(k1 = 7 * 10**8, k2=7 * 10**8)
car_vec_func = car_function_gen.car_vec_func
plot_all_variables_with_step(step, car_vec_func, r'$k_1 = k_2 = 7.10^8$')
plt.show()

