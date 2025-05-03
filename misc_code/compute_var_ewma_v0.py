
import matplotlib.pyplot as plt
import numpy as np

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


#    return result

def compute_etl(var_ref, return_list, n_steps):

    min_var = min(return_list)

    n_bins_sum = 0
    etl_sum_tmp = 0.0

    list_to_plot = []
    delta_var = (var_ref - min_var) / n_steps
    for i in range(0, n_steps):
        var_left = min_var + i * delta_var
        var_right = min_var + (i + 1) * delta_var
        var_mean = (var_left + var_right) / 2.0

        out_histo = np.histogram(return_list, bins=[var_left, var_right])

        #print('out_histo: ', out_histo)
        n_bins_ = out_histo[0][0]
        n_bins_sum = n_bins_sum + n_bins_
        etl_sum_tmp = n_bins_ * var_mean + etl_sum_tmp

        #list_to_plot.append(n_bins_)
    #plt.plot(list_to_plot)
    #plt.show()
    #FQ(888)

    etl_end = etl_sum_tmp / n_bins_sum

    return etl_end



def get_simulated_return_(time_horizon, dt, n_trj, sigma, mu, p_tau):

    N = round(time_horizon/ dt)
    t = np.linspace(0, time_horizon, N)

    s_list = []

    mu = mu - 0.005

    for i in range(0, n_trj):

        #W = np.random.standard_normal(size=N)
        W = mu  -sigma*np.random.uniform(0, +1.0, N)


        W_n = []

        n_steps  = 10
        for i in range(0, len(W)):

            if (i < n_steps):
                w_n = W[i]
            else:
                w_m = 0
                for j in range(0, n_steps):

                    w_new = W[i-j]
                    w_m = w_m + W[i-j]

                w_m = w_m/n_steps
                w_n = p_tau*w_m + (1 - p_tau)*W[i]

            W_n.append(w_n)

        r_sum = np.sum(W_n)/N
        s_return = r_sum

        s_list.append(s_return)

    return  s_list


def get_simulated_return(time_horizon, dt, n_trj, sigma, mu, p_tau):

    N = round(time_horizon/ dt)
    t = np.linspace(0, time_horizon, N)

    s_list = []


    for i in range(0, n_trj):

        #W = np.random.standard_normal(size=N)
        W = mu  -sigma*np.random.uniform(0, +1.0, N)


        W_n = []

        n_steps  = 10
        for i in range(0, len(W)):

            if (i < n_steps):
                w_n = W[i]
            else:
                w_m = 0
                for j in range(0, n_steps):

                    w_new = W[i-j]
                    w_m = w_m + W[i-j]
                    w_old = w_new
                w_m = w_m/n_steps
                w_n = p_tau*w_m + (1 - p_tau)*W[i]

            W_n.append(w_n)

        r_sum = np.sum(W_n)/N
        s_return = r_sum

        s_list.append(s_return)

    return  s_list


def compute_log(sp_values, time_step):

    log_return = []
    ln = len(sp_values)

    for i in range(0, ln-1, time_step):
        log_r_tmp = np.log(sp_values[i + 1]) - np.log(sp_values[i])

        if (np.abs(log_r_tmp)<0.0001):

            continue
        else:

            log_return.append(log_r_tmp)

    return log_return


def compute_log_tw(sp_values, n_start, n_end):

    log_return = []
    ln = len(sp_values)

    for i in range(n_start, n_end, 1):
        log_r_tmp = np.log(sp_values[i + 1]) - np.log(sp_values[i])

        if (np.abs(log_r_tmp)<0.0001):

            continue
        else:

            log_return.append(log_r_tmp)

    return log_return

def compute_ewma(data_to_ewma, alpha):

    data_ewma = []
    data_ewma.append(data_to_ewma[0])
    ewma_o = data_to_ewma[0]

    for i in range(1, ln-1):
        var_n = data_to_ewma[i]
        ewma_n = alpha*var_n + (1-alpha)*ewma_o
        ewma_o = ewma_n
        data_ewma.append(ewma_n)

    return data_ewma


if __name__ == '__main__':

    import pandas as pd

    file_data = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\week_10\data\sp_data.xlsx"
    xx = pd.read_excel(open(file_data, 'rb'))

    ln = len(xx)
    n_pts = ln

    xx = xx[ln - n_pts:ln]

    xx = xx.reset_index()

    sp_values_old = xx['SP_500_IDX']

    alpha = 0.01
    sp_values = compute_ewma(sp_values_old, alpha)
    ln = len(sp_values)


    plt.plot(sp_values_old)
    plt.plot(sp_values)
    plt.show()

    #FQ(99)
    sp_end   = sp_values[ln - 1]
    sp_start = sp_values[0]

    log_return  =[]
    var_list = []

    cl = 0.01
    time_step = 10

    var_list1 = []
    var_list2 = []

    time_list = []

    tw1 = 100
    tw2 = 500

    for i in range(1, ln - tw2 -1):

        n_start  = i
        n_end   = i + tw2

        log_return1 = compute_log_tw(sp_values, n_start, n_end)
        log_return2 = compute_log_tw(sp_values_old, n_start, n_end)

        var_tmp1 = np.quantile(log_return1, cl, axis=0)
        var_tmp2 = np.quantile(log_return2, cl, axis=0)

        var_list1.append(var_tmp1)
        var_list2.append(var_tmp2)

        time_list.append(i)


    plt.plot(time_list, var_list2)
    plt.plot(time_list, var_list1)
    plt.legend(['PLAIN', 'EWMA'])
    plt.xlabel('N. days', fontsize=14)
    plt.ylabel('Var level', fontsize=14)


    plt.show()
    FQ(1888)
