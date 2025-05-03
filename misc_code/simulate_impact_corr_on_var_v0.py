
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

if __name__ == '__main__':



    sigma = 0.35
    mu = 0.0
    time_horizon = 5


    S0    = 100
    n_trj = 10000

    q = 0.0
    dt = 0.1

    p_tau_1 = 0
    p_tau_2 = 1

    s_list_1 = get_simulated_return(time_horizon, dt, n_trj, sigma, mu, p_tau_1)
    s_list_2 = get_simulated_return_(time_horizon, dt, n_trj, sigma, mu, p_tau_2)

    #FQ(88)

    n_alfa_steps = 200
    n_es_steps = 20

    var_list_1 = []
    var_list_2 = []

    alpha_list = []

    for i in range(0, n_alfa_steps):
        alpha_tmp = 0.001 + i*0.005

        var_tmp_1 = np.quantile(s_list_1, alpha_tmp, axis=0)
        var_tmp_2 = np.quantile(s_list_2, alpha_tmp, axis=0)

        var_list_1.append(var_tmp_1)
        var_list_2.append(var_tmp_2)
        alpha_list.append(alpha_tmp)


    plt.plot(alpha_list, var_list_1)
    plt.plot(alpha_list, var_list_2)

    plt.legend(['VaR sigma=%s mu= %s T= %s'%(sigma, mu, time_horizon), 'VaR with Corr sigma=%s mu= %s T= %s'%(sigma, mu, time_horizon)])

    plt.xlabel('Alpha level', fontsize=14)
    plt.ylabel('VaR[return]', fontsize=14)
    #plt.xlim([0, 0.4])
    #plt.ylim([-1.0, 0.0])

    plt.show()
    plt.clf()


    plt.hist(s_list_1, bins=int(n_trj/10))

    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('N. entries', fontsize=14)
    plt.title('Underlying distribution (S)', fontsize=14)
    plt.show()





