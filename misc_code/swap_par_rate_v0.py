import matplotlib.pyplot as plt
import numpy as np
import sys


def swap_rate(T_in, pay_freq, zc_rate_time, zc_rate_valu):

    N = int(T_in/pay_freq)
    t_pay = np.linspace(0, T_in, N+1)

    ann = 0
    z_o = 1.0
    delta_t = 0.5
    for i in range(1, len(t_pay)):
        t = t_pay[i]
        zc_rate_t = np.interp(t, zc_rate_time, zc_rate_valu)
        z_n = z_o/(1.0 + zc_rate_t*delta_t)

        z_o = z_n
        ann = z_n*delta_t + ann

    sw_rate = (1.0 -z_n)/ann

    return sw_rate



if __name__ == '__main__':



    libor_rate_value = [0.01, 0.015, 0.03,  0.03,  0.03, 0.032, 0.035]
    libor_rate_times = [0.0,    1.0,  2.0,   2.5,   3.0,   5.0,  10.0]

    t = 0
    pay_freq = 1.0

    T_mat = 10
    N = int(T_mat/pay_freq)
    t_mat_list = np.linspace(0, T_mat, N+1)


    swp_rate_t_list = []
    t_mat_list_to_plot = []

    for i in range(1, len(t_mat_list)):
        t_mat = t_mat_list[i]
        swp_rate_t = swap_rate(t_mat, pay_freq, libor_rate_times, libor_rate_value)
        swp_rate_t_list.append(swp_rate_t)
        t_mat_list_to_plot.append(t_mat_list[i])


    plt.plot(t_mat_list_to_plot, swp_rate_t_list, '.-')
    plt.plot(libor_rate_times, libor_rate_value, '.-')

    plt.legend(['Swap rate freq. 6M', 'Risk free rate'])
    plt.xlabel('Maturity [year]', fontsize=14)
    plt.ylabel('Level of interest rate ', fontsize=14)
    plt.title('Swap vs risk free rate ', fontsize=14)

    plt.show()
    sys.exit()
