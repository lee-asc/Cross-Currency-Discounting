
import matplotlib.pyplot as plt
import numpy as np

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def compute_es(var_ref, return_list, n_steps):

    min_var = min(return_list)

    n_bins_sum = 0
    es_sum_tmp = 0.0
    delta_var = (var_ref - min_var) / n_steps
    for i in range(0, n_steps):
        var_left = min_var + i * delta_var
        var_right = min_var + (i + 1) * delta_var
        var_mean = (var_left + var_right) / 2.0

        out_histo = np.histogram(return_list, bins=[var_left, var_right])
        n_bins_ = out_histo[0][0]
        n_bins_sum = n_bins_sum + n_bins_
        es_sum_tmp = n_bins_ * var_mean + es_sum_tmp

    es_end = es_sum_tmp / n_bins_sum

    return es_end


if __name__ == '__main__':



    #------- model parameters
    sigma_1 = 0.10
    sigma_2 = 0.10
    sigma_3 = 0.20

    mu_1 = 0.01
    mu_2 = 0.05
    mu_3 = 0.1

    t_1 = 5
    t_2 = 5
    t_3 = 5


    S0    = 100
    #------- payoff parameters
    T     = 5;
    #r = mu
    #------- MC parameters
    n_trj = 10000

    q = 0.0
    dt = 0.01

    N1 = round(t_1/ dt)
    N2 = round(t_2/ dt)
    N3 = round(t_3/ dt)

    t1 = np.linspace(0, t_1, N1)
    t2 = np.linspace(0, t_2, N2)
    t3 = np.linspace(0, t_3, N3)

    s1_list = []
    s2_list = []
    s3_list = []


    for i in range(0, n_trj):

        W_1 = np.random.standard_normal(size=N1)
        W_1 = np.cumsum(W_1) * np.sqrt(dt)  ### standard brownian motion ###

        W_2 = np.random.standard_normal(size=N2)
        W_2 = np.cumsum(W_2) * np.sqrt(dt)  ### standard brownian motion ###

        W_3 = np.random.standard_normal(size=N3)
        W_3 = np.cumsum(W_3) * np.sqrt(dt)  ### standard brownian motion ###


        X_1 = (mu_1 - 0.5 * sigma_1**2) * t1 + sigma_1 * W_1
        X_2 = (mu_2 - 0.5 * sigma_2**2) * t2 + sigma_2 * W_2
        X_3 = (mu_3 - 0.5 * sigma_3**2) * t3 + sigma_3 * W_3

        S_1 = S0 * np.exp(X_1)  ### geometric brownian motion ###
        S_2 = S0 * np.exp(X_2)  ### geometric brownian motion ###
        S_3 = S0 * np.exp(X_3)  ### geometric brownian motion ###

        #p_0_t_1 = np.exp(-r*T)

        s1_return = (S_1[N1-1] - S0)/S0
        s2_return = (S_2[N2-1] - S0)/S0
        s3_return = (S_3[N3-1] - S0)/S0

        s1_list.append(s1_return)
        s2_list.append(s2_return)
        s3_list.append(s3_return)


    var1_list = []
    var2_list = []
    var3_list = []

    alpha_list = []
    n_alpha_steps = 200

    for i in range(0, n_alpha_steps):
        alpha_tmp = 0.001 + i*0.005

        var1_tmp = np.quantile(s1_list, alpha_tmp, axis=0)
        var2_tmp = np.quantile(s2_list, alpha_tmp, axis=0)
        var3_tmp = np.quantile(s3_list, alpha_tmp, axis=0)

        var1_list.append(var1_tmp)
        var2_list.append(var2_tmp)
        var3_list.append(var3_tmp)

        alpha_list.append(alpha_tmp)


    plt.plot(alpha_list, var1_list)
    plt.plot(alpha_list, var2_list)
    plt.plot(alpha_list, var3_list)
    plt.legend(['sigma=%s mu= %s T= %s'%(sigma_1, mu_1, t_1), 'sigma=%s mu= %s T= %s'%(sigma_2, mu_2, t_2), 'sigma=%s mu= %s T= %s'%(sigma_3, mu_3, t_3)])
    plt.xlabel('Alpha level', fontsize=14)
    plt.ylabel('VaR[return]', fontsize=14)
    plt.xlim([0, 0.1])
    plt.ylim([-0.75, 0.2])

    plt.show()

    plt.plot(alpha_list, var1_list)
    plt.plot(alpha_list, var2_list)
    plt.plot(alpha_list, var3_list)
    plt.legend(['sigma=%s mu= %s T= %s'%(sigma_1, mu_1, t_1), 'sigma=%s mu= %s T= %s'%(sigma_2, mu_2, t_2), 'sigma=%s mu= %s T= %s'%(sigma_3, mu_3, t_3)])
    plt.xlabel('Alpha level', fontsize=14)
    plt.ylabel('VaR[return]', fontsize=14)

    plt.show()


    plt.hist(s1_list, bins=int(n_trj/10))

    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('N. entries', fontsize=14)
    plt.title('Underlying distribution (S)', fontsize=14)
    plt.show()

    plt.hist(s2_list, bins=int(n_trj/10))

    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('N. entries', fontsize=14)
    plt.title('Underlying distribution (S)', fontsize=14)
    plt.show()

    plt.hist(s3_list, bins=int(n_trj/10))

    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('N. entries', fontsize=14)
    plt.title('Underlying distribution (S)', fontsize=14)
    plt.show()

    #plt.hist(call_payoff_list, bins=int(n_trj/10))

    #plt.xlabel('Stock level', fontsize=14)
    #plt.ylabel('N. entries', fontsize=14)
    #plt.title('Payoff distribution', fontsize=14)
    #plt.show()




    FQ(88)

