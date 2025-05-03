import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import sys


def random_weights(n):
    k = np.random.rand(n)
    return k / sum(k)


if __name__ == '__main__':


    file_data_eu = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\eu_data.xlsx"
    file_data_jp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\jp_data.xlsx"
    file_data_sp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\sp_data.xlsx"

    ts_jp = pd.read_excel(open(file_data_jp, 'rb'))
    ts_eu = pd.read_excel(open(file_data_eu, 'rb'))
    ts_sp = pd.read_excel(open(file_data_sp, 'rb'))

    dates_    = ts_sp['DATE']
    sp_values = ts_sp['SP_500_IDX']
    eu_values = ts_eu['MSCI_EUROPE']
    jp_values = ts_jp['MSCI_JAPAN']

    close_price = pd.concat([dates_, sp_values, eu_values, jp_values], axis=1)
    close_price = close_price.set_index('DATE')

    years = 45

    n_samples = 10000



    # ---Compute Daily Log returns ----------------------
    log_return_n = np.log(close_price) - np.log(close_price.shift(1))

    # ---Compute Returns on full time serie period -------------------
    r_yy = (np.log(close_price.iloc[-1]) - np.log(close_price.iloc[0]))/years


    # ---Compute Covariance ----------------------
    cov = log_return_n.cov()


    exp_return1 = []
    exp_return2 = []

    sigma1 = []
    sigma2 = []


    corr_param = 0.6
    for _ in range(n_samples):

        w = random_weights(3)

        exp_return1.append(w @ r_yy.T)
        exp_return2.append(w @ r_yy.T)

        var_1 = (w.T @ cov @ w)
        var_2 = (w.T @ (corr_param*cov) @ w)

        sigma1.append(np.sqrt(var_1))
        sigma2.append(np.sqrt(var_2))

    plt.plot(sigma1, exp_return1, 'ro', alpha=0.8)
    plt.plot(sigma2, exp_return2, 'go', alpha=0.8)
    plt.legend(['High correlation', 'Low correlation (%s)'%corr_param])
    plt.xlabel('Risk [$\sigma$]', fontsize=14)
    plt.ylabel('Returns ', fontsize=14)
    plt.title('20k weights configurations for a ptf with 3 stocks ', fontsize=14)

    plt.show()
