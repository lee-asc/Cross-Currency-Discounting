import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import scipy.stats as scpy
import sys

#import sympy as sy
#from sympy.stats import Normal, cdf
#from sympy import init_printing



def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


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
        delta_g = scpy.norm.cdf(d1, 0.0, 1.0)
        gamma_g = scpy.norm.pdf(d1, 0.0, 1.0)/(S*sigma*T)
        vega_g  = scpy.norm.pdf(d1, 0.0, 1.0)*S*np.sqrt(T)
        rho_g   = K*scpy.norm.pdf(d2, 0.0, 1.0)*T*np.exp(-r*T)
        volga_g = vega_g*d1*d2/sigma

    if option == 'put':
        result  = (K * np.exp(-r * T) * scpy.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * scpy.norm.cdf(-d1, 0.0, 1.0))
        delta_g = 1.0 - scpy.norm.cdf(d1, 0.0, 1.0)
        gamma_g = scpy.norm.pdf(d1, 0.0, 1.0)/(S*sigma*T)
        vega_g  = scpy.norm.pdf(d1, 0.0, 1.0)*S*np.sqrt(T)
        rho_g   = -K*scpy.norm.pdf(-d2, 0.0, 1.0)*T*np.exp(-r*T)
        volga_g = vega_g*d1*d2/sigma

    return [result, delta_g, gamma_g, vega_g, rho_g, volga_g]


if __name__ == '__main__':

    S = 100
    K = 100
    r = 0.01
    q = 0.0
    T = 2.0
    sigma = 0.15

    n_step = 100
    s_range = np.linspace(10, 200, n_step + 1)

    # --------------------------------------------------------
    # ------ IMPACT OF VOLATILITY ------------------------------
    # --------------------------------------------------------

    call_price_range = []
    vol_range = [0.001, 0.1, 0.35, 0.70]
    legend_list = []


    for vol_tmp in vol_range:
        call_price_range = []

        for s_tmp in s_range:

            call_price_val_tmp = euro_vanilla_dividend(s_tmp, K, T, r, q, vol_tmp, option='call')
            call_price_range.append(call_price_val_tmp[0])


        plt.plot(s_range, call_price_range, '-')
        legend_list.append(('Sigma level: %s'%str(vol_tmp)))

    plt.legend(legend_list)
    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('Call price', fontsize=14)
    plt.title('Impact of volatility (sigma) on Call price', fontsize=14)

    plt.show()

    #FQ(778)


    #--------------------------------------------------------
    #------ IMPACT OF MATURITY ------------------------------
    #--------------------------------------------------------



    S = 100
    K = 100
    r = 0.01
    q = 0.0
    T = 2.0
    sigma = 0.15

    n_step = 100
    s_range = np.linspace(10, 200, n_step + 1)

    call_price_range = []
    mat_range = [0.5, 2.0, 5.0, 10.0]
    legend_list = []

    for mat_tmp in mat_range:
        call_price_range = []
        for s_tmp in s_range:

            call_price_val_tmp = euro_vanilla_dividend(s_tmp, K, mat_tmp, r, q, sigma, option='call')
            call_price_range.append(call_price_val_tmp[0])


        plt.plot(s_range, call_price_range, '-')
        legend_list.append(('Mat level: %sY'%str(mat_tmp)))

    plt.legend(legend_list)
    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('Call price', fontsize=14)
    plt.title('Impact of Maturity on Call price', fontsize=14)

    plt.show()



    #--------------------------------------------------------
    #------ IMPACT OF STRIKE ------------------------------
    #--------------------------------------------------------


    S = 100
    K = 100
    r = 0.01
    q = 0.0
    T = 2.0
    sigma = 0.15

    n_step = 100
    s_range = np.linspace(10, 200, n_step + 1)

    call_price_range = []
    strike_range = [0.5, 50, 100, 200]
    legend_list = []

    for strike_tmp in strike_range:
        call_price_range = []
        for s_tmp in s_range:

            call_price_val_tmp = euro_vanilla_dividend(s_tmp, strike_tmp, T, r, q, sigma, option='call')
            call_price_range.append(call_price_val_tmp[0])


        plt.plot(s_range, call_price_range, '-')
        legend_list.append(('Strike level: %s'%str(strike_tmp)))

    plt.legend(legend_list)
    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('Call price', fontsize=14)
    plt.title('Impact of Strike on Call price', fontsize=14)

    plt.show()




    #--------------------------------------------------------
    #------ IMPACT OF Interest rate ------------------------------
    #--------------------------------------------------------


    S = 100
    K = 100
    r = 0.01
    q = 0.0
    T = 2.0
    sigma = 0.15

    n_step = 100
    s_range = np.linspace(10, 200, n_step + 1)

    call_price_range = []
    #r_range = [0.0, 0.01, 0.02, 0.03, 0.05]
    r_range = [0.0, 0.03, 0.05, 0.1]

    legend_list = []

    for r_tmp in r_range:
        call_price_range = []
        for s_tmp in s_range:

            call_price_val_tmp = euro_vanilla_dividend(s_tmp, K, T, r_tmp, q, sigma, option='call')
            call_price_range.append(call_price_val_tmp[0])


        plt.plot(s_range, call_price_range, '-')
        legend_list.append(('Interst rate: %s'%str(r_tmp)))

    plt.legend(legend_list)
    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('Call price', fontsize=14)
    plt.title('Impact of Interest rates on Call price', fontsize=14)

    plt.show()

    FQ(99)

