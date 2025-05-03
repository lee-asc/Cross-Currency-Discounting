import pandas_datareader as pdr
import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yfin


import sys



def random_weights(n):
    k = np.random.rand(n)
    w_ = k / sum(k)
    w_list = []
    for i in range(0, len(w_)):
        w_list.append(w_[i])

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
if __name__ == '__main__':


    yfin.pdr_override()

    tickers = ['AAPL', 'MSFT', 'IBM', '^GSPC', 'META', 'RACE', 'XOM']

    # '^GSPC' indicate the SP500
    start   = dt.datetime(2015, 12, 1)
    end     = dt.datetime(2021, 1, 1)

    data = pdr.get_data_yahoo(tickers, start, end)

    data = data['Adj Close']

    close_price = data
    #close_price = close_price.set_index('Date')


    years = 5
    log_returns = np.log(data / data.shift())
    r_yy = (np.log(close_price.iloc[-1]) - np.log(close_price.iloc[0]))/years


    print('r_yy: ', r_yy)
    #sys.exit()
    plt.plot(log_returns)
    plt.ylabel('Log-return')
    plt.xlabel('Time [years]')
    plt.legend(tickers)

    plt.show()

    cov = log_returns.cov()

    w0 = random_weights(7)
    var_0 = (w0.T @ cov @ w0)
    r_0 = w0 @ r_yy.T

    r_target_list = [0.05, 0.07, 0.12, 0.15, 0.20, 0.25, 0.3]
    w_bnd = [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

    r_opt_list = []
    var_opt_list = []

    for i in range(0, len(r_target_list)):


        r_target = r_target_list[i]
        print('Optimize r-target: ', r_target)

        arguments = [r_yy, r_target]
        cons = [{'type': 'eq', 'fun': cons_w}, {'type': 'eq', 'fun': cons_r, 'args': arguments}]

        ff = optimize.minimize(loss_var, w0, cov, bounds=w_bnd, constraints=cons)
        #ff = optimize.minimize(loss_var, w0, cov, constraints=cons)

        w_opt = ff.x
        var_opt = (w_opt.T @ cov @ w_opt)
        r_opt = (w_opt @ r_yy.T)

        print('w_opt: ', w_opt)
        r_opt_list.append(r_opt)
        var_opt_list.append(var_opt)


    plt.plot(var_opt_list, r_opt_list, 'ro--')
    plt.ylabel('Return')
    plt.xlabel('Var')

    plt.show()

