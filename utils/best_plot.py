import numpy as np

def find_best_plot_ten_expoent(*ys_values):
    ys_max_values = []
    for y_values in ys_values:
        ys_max_values.append(np.max(np.abs(y_values)))
    
    absolut_max = max(ys_max_values)

    ys_ten_expoents = []
    for y_max_value in ys_max_values:
        round_down_ten_expoent = int(np.log10(absolut_max / y_max_value))
        round_up_ten_expoent = int(np.log10(absolut_max / y_max_value)) + 1

        round_down_diff = abs(y_max_value * 10**round_down_ten_expoent - absolut_max)
        round_up_diff = abs(y_max_value * 10**round_up_ten_expoent - absolut_max)

        if round_down_diff < round_up_diff:
            ys_ten_expoents.append(round_down_ten_expoent)
        else:
            ys_ten_expoents.append(round_up_ten_expoent)

    return ys_ten_expoents

