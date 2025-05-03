
import matplotlib.pyplot as plt
import numpy as np

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()
import pandas as pd


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


    file_data = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\sp_data.xlsx"
    xx = pd.read_excel(open(file_data, 'rb'))

    ln = len(xx)
    n_pts = ln

    xx = xx[ln - n_pts:ln]
    xx = xx.reset_index()

    sp_values = xx['SP_500_IDX']
    ln = len(sp_values)

    sp_end   = sp_values[ln - 1]
    sp_start = sp_values[0]

    cl        = 0.01
    time_list = []

    tw_var_ref = 1000

    n_start = 0
    n_end   = tw_var_ref

    log_return_for_var = compute_log_tw(sp_values, n_start, n_end)
    var_limit = np.quantile(log_return_for_var, cl, axis=0)

    idx_var_sum      = 0
    time_list        = []
    var_limit_list   = []
    log_return_list  = []
    obs_var_sum_list = []
    exp_idx_var_sum_list = []
    exp_idx_var_sum_p_list = []
    exp_idx_var_sum_m_list = []
    exp_idx_var_sum = 0


    for i in range(1, ln - tw_var_ref - 1):

        log_r_tmp = np.log(sp_values[i + 1]) - np.log(sp_values[i])

        if (log_r_tmp < var_limit):
            idx_var = 1
        else:
            idx_var = 0

        idx_var_sum = idx_var_sum + idx_var
        exp_idx_var_sum = exp_idx_var_sum + 1
        exp_val = exp_idx_var_sum*cl
        exp_err = np.sqrt(exp_idx_var_sum*cl*(1-cl))
        exp_val_p = exp_val + exp_err
        exp_val_m = exp_val - exp_err

        time_list.append(i)

        var_limit_list.append(var_limit)
        log_return_list.append(log_r_tmp)

        exp_idx_var_sum_list.append(exp_val)
        exp_idx_var_sum_p_list.append(exp_val_p)
        exp_idx_var_sum_m_list.append(exp_val_m)

        obs_var_sum_list.append(idx_var_sum)



    plt.plot(time_list, var_limit_list, '--r')
    plt.plot(time_list, log_return_list)
    plt.legend(['Var limit','Log-return'])
    plt.xlabel('N. days', fontsize=14)
    plt.ylabel('Var level', fontsize=14)


    plt.show()

    plt.plot(time_list, exp_idx_var_sum_p_list, '--g')
    plt.plot(time_list, exp_idx_var_sum_list, 'b')
    plt.plot(time_list, exp_idx_var_sum_m_list, '--k')
    plt.plot(time_list, obs_var_sum_list, 'or')
    plt.legend(['Exp + err excees.', 'Exp excees.', 'Exp - err excees.','Obs. excees.'])
    plt.xlabel('N. days', fontsize=14)
    plt.ylabel('N. events', fontsize=14)

    plt.show()
