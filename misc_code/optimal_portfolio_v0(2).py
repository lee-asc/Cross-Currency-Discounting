import pandas_datareader as pdr
import datetime as dt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import sys

def random_weights(n):
    k = np.random.rand(n)
    w_ = k / sum(k)

    return w_


from scipy import optimize
from scipy.optimize import minimize
from scipy.optimize import fmin


def loss_var(w_list, cov):


    var = (w_list.T @ cov @ w_list)


    return var

def cons_w(w_list):

    w_sum = 0
    for i in range(0, len(w_list)):
        w_sum = w_sum + w_list[i]

    w_sum = w_sum - 1
    return w_sum

def cons_r(w_list, r_yy, r_target):

    r_0 = w_list @ r_yy.T

    return r_0 - r_target


import numpy as np
import matplotlib.pyplot as plt
import sys

from pandas_datareader import data as pdr
import yfinance as yfin


if __name__ == '__main__':

    yfin.pdr_override()


    tickers = ['AAPL', 'MSFT', 'IBM', '^GSPC', 'META', 'RACE', 'XOM']

    # '^GSPC' indicate the SP500
    start   = dt.datetime(2015, 12, 1)
    end     = dt.datetime(2021, 1, 1)

    data = pdr.get_data_yahoo(tickers, start, end)

    data = data['Adj Close']

    close_price = data
    # compute log-return

    years = 5
    log_returns = np.log(data) - np.log(data.shift())
    r_yy = (np.log(close_price.iloc[-1]) - np.log(close_price.iloc[0]))/years


    plt.plot(log_returns)
    plt.ylabel('Log-return')
    plt.xlabel('Time [years]')
    plt.legend(tickers)

    plt.show()

    w = random_weights(7)

    # -Compute covariance of generic ptf
    cov = log_returns.cov()

    # -Compute risk and return of generic ptf
    w0 = random_weights(7)
    var_0 = (w0.T @ cov @ w0)
    r_0 = w0 @ r_yy.T

    r_target = 0.1

    arguments = [r_yy, r_target]
    cons = [{'type': 'eq', 'fun': cons_w}, {'type': 'eq', 'fun': cons_r, 'args': arguments}]

    w_bnd = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    ff = optimize.minimize(loss_var, w0, cov, bounds=w_bnd, constraints=cons)

    w_opt = ff.x
    var_opt = (w_opt.T @ cov @ w_opt)
    r_opt = (w_opt @ r_yy.T)




    print('w0: ', w0)
    print('var_0: %.4f'%var_0)
    print('r_0: %.4f'%r_0)

    print('=====================')
    print('w_opt: ', w_opt)
    print('var_opt: %.4f'%var_opt)
    print('r_opt: %.4f'%r_opt)

    sys.exit()
