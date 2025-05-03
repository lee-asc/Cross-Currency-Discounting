import numpy as np
import matplotlib.pyplot as plt
#import pandas_datareader as pdr
#import pandas as pd
#import datetime as dt
#from dateutil.relativedelta import relativedelta
import scipy.stats as scpy
import matplotlib.pyplot as plt
import numpy as np

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def random_weights(n):
    k = np.random.rand(n)
    return k / sum(k)

def euro_vanilla_dividend(S, K, T, r, q, sigma, option='call'):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # q: rate of continuous dividend paying asset
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    if option == 'call':
        result = (S * np.exp(-q * T) * scpy.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * scpy.norm.cdf(d2, 0.0, 1.0))
    if option == 'put':
        result = (K * np.exp(-r * T) * scpy.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * scpy.norm.cdf(-d1, 0.0, 1.0))

    return result


if __name__ == '__main__':



    #------- model parameters
    mu    = 0.05;
    sigma = 0.015
    S0    = 100
    #------- payoff parameters
    T     = 5;
    strike = S0
    r = mu
    #------- MC parameters
    n_trj = 5000

    q = 0.0
    dt = 0.01
    N = round(T/ dt)
    t = np.linspace(0, T, N)

    s_list = []
    call_price_list = []

    call_price_sum = 0.0

    for i in range(0, n_trj):

        W_1 = np.random.standard_normal(size=N)
        W_1 = np.cumsum(W_1) * np.sqrt(dt)  ### standard brownian motion ###

        X_1 = (mu - 0.5 * sigma**2) * t + sigma * W_1

        S = S0 * np.exp(X_1)  ### geometric brownian motion ###

        p_0_t_1 = np.exp(-r*T)

        call_payoff = np.maximum(S[N-1]  - strike, 0)
        call_price  = p_0_t_1 * call_payoff

        call_price_list.append(call_price)

        s_list.append(S[N-1])
        call_price_sum = call_price_sum + call_price

    call_price_mc = call_price_sum/n_trj

    call_price_anlytic = euro_vanilla_dividend(S0, strike, T, r, q, sigma, option='call')
    print('call_price_mc: %.2f'%(call_price_mc))
    print('call_price_anlytic: %.2f'%(call_price_anlytic))

    plt.hist(s_list, bins=int(n_trj/10))

    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('N. entries', fontsize=14)
    plt.title('Underlying distribution (S)', fontsize=14)
    plt.show()

    plt.hist(call_price_list, bins=int(n_trj/10))

    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('N. entries', fontsize=14)
    plt.title('Payoff distribution', fontsize=14)
    plt.show()





