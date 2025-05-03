
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys



# Press the green button in the gutter to run the script.

def compute_ptf_var(sp_values, jp_values, eu_values, ln):

    ptf_value = 1.0 * sp_values + 1.0 * jp_values + 1.0 * eu_values

    w_sp_0 = sp_values[0] / ptf_value[0]
    w_jp_0 = jp_values[0] / ptf_value[0]
    w_eu_0 = eu_values[0] / ptf_value[0]

    return_sp = []
    return_jp = []
    return_eu = []
    return_ptf = []

    return_ptf_sum = 0.0

    for i in range(0, ln - 1):

        return_sp_tmp = (sp_values[i + 1] - sp_values[i]) / sp_values[i]
        return_jp_tmp = (jp_values[i + 1] - jp_values[i]) / jp_values[i]
        return_eu_tmp = (eu_values[i + 1] - eu_values[i]) / eu_values[i]
        return_ptf_tmp = (ptf_value[i + 1] - ptf_value[i]) / ptf_value[i]

        #ptf_tot = sp_values[i] + jp_values[i] + eu_values[i]

        return_eu.append(return_eu_tmp)
        return_jp.append(return_jp_tmp)
        return_sp.append(return_sp_tmp)
        return_ptf.append(return_ptf_tmp)


    # print('return_ptf_tmp: ', return_ptf)
    #ptf_mean = np.asarray(return_ptf).mean()
    #ptf_var = np.asarray(return_ptf).var()

    return_eu_a = np.asarray(return_eu)
    return_jp_a = np.asarray(return_jp)
    return_sp_a = np.asarray(return_sp)

    return_eu_var = return_eu_a.var()
    return_jp_var = return_jp_a.var()
    return_sp_var = return_sp_a.var()

    x = np.array([return_eu_a, return_jp_a, return_sp_a])
    w = np.array([w_eu_0, w_jp_0, w_sp_0])
    portfolio_variance = np.dot(w, np.dot(np.cov(x), w.T))

    w_sum_var = 1.0/3.0*return_eu_var + 1.0/3.0*return_jp_var + 1.0/3.0*return_sp_var
    return w_sum_var, portfolio_variance

if __name__ == '__main__':


    file_data_eu = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\eu_data.xlsx"
    file_data_jp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\jp_data.xlsx"
    file_data_sp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\sp_data.xlsx"

    ts_jp = pd.read_excel(open(file_data_jp, 'rb'))
    ts_eu = pd.read_excel(open(file_data_eu, 'rb'))
    ts_sp = pd.read_excel(open(file_data_sp, 'rb'))

    sp_values = ts_sp['SP_500_IDX']
    eu_values = ts_eu['MSCI_EUROPE']
    jp_values = ts_jp['MSCI_JAPAN']

    ptf_var_o_list = []
    ptf_var_c_list = []
    n_obs = []

    for i in range(10, 2400,10):

        w_ptf_sum_var, ptf_var_mat = compute_ptf_var(sp_values, jp_values, eu_values, i)
        ptf_var_o_list.append(np.sqrt(w_ptf_sum_var))
        ptf_var_c_list.append(np.sqrt(ptf_var_mat))
        n_obs.append(i)


    plt.plot(n_obs, ptf_var_o_list)
    plt.plot(n_obs, ptf_var_c_list)
    plt.legend(['Weighted sum of Var', 'Portfolio Var'])

    plt.show()

    print('XXXXXXXXX')
