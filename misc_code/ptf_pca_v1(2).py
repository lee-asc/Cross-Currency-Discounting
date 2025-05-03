import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    file_data_eu = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\eu_data.xlsx"
    file_data_jp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\jp_data.xlsx"
    file_data_sp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\sp_data.xlsx"


    data_sp = pd.read_excel(open(file_data_sp, 'rb'))
    data_jp = pd.read_excel(open(file_data_jp, 'rb'))
    data_eu = pd.read_excel(open(file_data_eu, 'rb'))

    sp_values = data_sp['SP_500_IDX']
    eu_values = data_eu['MSCI_EUROPE']
    jp_values = data_jp['MSCI_JAPAN']

    ln = len(sp_values)

    ptf_value = 1.0 * sp_values + 1.0 * jp_values + 1.0 * eu_values


    w_sp_0 = sp_values[0] / ptf_value[0]; w_jp_0 = jp_values[0] / ptf_value[0]; w_eu_0 = eu_values[0] / ptf_value[0]

    return_sp = []
    return_jp = []
    return_eu = []
    return_ptf = []

    return_ptf_sum = 0.0

    # COMPUTE RETURN-----------
    for i in range(0, ln - 1):

        return_sp_tmp = np.log(sp_values[i + 1]) - np.log(sp_values[i])
        return_jp_tmp = np.log(jp_values[i + 1]) - np.log(jp_values[i])
        return_eu_tmp = np.log(eu_values[i + 1]) - np.log(eu_values[i])
        return_ptf_tmp = np.log(ptf_value[i + 1]) - np.log(ptf_value[i])

        ptf_tot = sp_values[i] + jp_values[i] + eu_values[i]

        return_eu.append(return_eu_tmp)
        return_jp.append(return_jp_tmp)
        return_sp.append(return_sp_tmp)
        return_ptf.append(return_ptf_tmp)

    return_eu_a = np.asarray(return_eu)
    return_jp_a = np.asarray(return_jp)
    return_sp_a = np.asarray(return_sp)

    eu_std = np.sqrt(np.var(return_eu_a))
    sp_std = np.sqrt(np.var(return_sp_a))
    jp_std = np.sqrt(np.var(return_jp_a))

    # Original vector of return
    x = np.array([return_eu_a, return_jp_a, return_sp_a])

    #print('x: ', x)
    #sys.exit()
    # Original weight

    w = np.array([w_eu_0, w_jp_0, w_sp_0])

    ptf_corr = np.corrcoef(x)


    std_vec = [eu_std, jp_std, sp_std]
    D = np.eye(3)
    D = D*std_vec

    #sys.exit()
    #--- VARCOV - matrix
    #--------------------
    V = D @ ptf_corr @ D

    #--- VAR value of ---
    var     = w @ V @ w.T

    #--- Eigenvalues/Eigen vectors decomposition ---
    ll, W = np.linalg.eigh(ptf_corr)

    print('eigv: ', ll)

    N = 3
    ind = sorted(range(N), key=lambda x: ll[x], reverse=True)

    lambda_eigv = ll[ind]
    #LL          = np.diag(lambda_eigv)

    print('sorted_eigv =', lambda_eigv)
    print('%_sorted_eigv =', lambda_eigv/lambda_eigv.sum())

    # P (2404,3)

    print('x size: ', np.size(W, axis = 1))

    "===== compute principal components ========"
    P   = x.T @ W
    P_n = P[:,1:] # WE CONSIDER JUST 1 - 3 COMPONENTS!!
    print('N. components P old: ', (P.shape))
    print('N. components P new: ', (P_n.shape))

    "====== compute new returns ======="
    x_n = P_n @ W[:,1:].T #3


    ptf_var_n = np.dot(w, np.dot(np.cov(x_n.T), w.T))
    print('ptf_sigma_n: ',np.sqrt(ptf_var_n))
    print('ptf_sigma_o: ', np.sqrt(var))


    sys.exit()
