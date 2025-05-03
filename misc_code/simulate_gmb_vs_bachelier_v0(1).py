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



    # simulation parameters
    n_trj = 50 # n. trajectories
    maturity = 1 # years
    n_time_steps = maturity*365
    dt = 1 # 1 day

    # model parameters
    mu = 0.01
    sigma1 = 0.1
    sigma2 = 10
    S0 = 100


    # initialize variables
    x_o = S0*np.ones((1, n_trj))
    x_o2 = S0*np.ones((1, n_trj))
    x_n = np.ones((n_time_steps, n_trj))
    x_n2 = np.ones((n_time_steps, n_trj))


    for i in range(0, n_time_steps):

        W = np.random.standard_normal(size=n_trj)
        Wt = W*np.sqrt(dt)  ### standard brownian motion ###

        x_n[i,:] = x_o + x_o*(mu*dt + sigma1*Wt)
        x_n2[i,:] = x_o2 + x_o2*(mu*dt) + sigma2*Wt
        #x_n[i,:] = x_o +(mu*dt + sigma*W)

        x_o = x_n[i,:]
        x_o2 = x_n2[i,:]


    plt.plot(x_n)
    plt.xlabel('Time [days]', fontsize=14)
    plt.ylabel('Price level', fontsize=14)
    plt.title('GBM Price evolution', fontsize=14)
    plt.show()

    plt.plot(x_n2)
    plt.xlabel('Time [days]', fontsize=14)
    plt.ylabel('Price level', fontsize=14)
    plt.title('Bachelier Price evolution', fontsize=14)
    plt.show()





    FQ(881)

