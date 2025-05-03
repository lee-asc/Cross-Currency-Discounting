import numpy as np
import matplotlib.pyplot as plt
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
        price = (S * np.exp(-q * T) * scpy.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * scpy.norm.cdf(d2, 0.0, 1.0))
        delta_g = scpy.norm.cdf(d1, 0.0, 1.0)
        gamma_g = scpy.norm.pdf(d1, 0.0, 1.0)/(S*sigma*T)
        vega_g  = scpy.norm.pdf(d1, 0.0, 1.0)*S*np.sqrt(T)
        rho_g   = K*scpy.norm.pdf(d2, 0.0, 1.0)*T*np.exp(-r*T)
        volga_g = vega_g*d1*d2/sigma

    if option == 'put':
        price  = (K * np.exp(-r * T) * scpy.norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * scpy.norm.cdf(-d1, 0.0, 1.0))
        delta_g = 1.0 - scpy.norm.cdf(d1, 0.0, 1.0)
        gamma_g = scpy.norm.pdf(d1, 0.0, 1.0)/(S*sigma*T)
        vega_g  = scpy.norm.pdf(d1, 0.0, 1.0)*S*np.sqrt(T)
        rho_g   = -K*scpy.norm.pdf(-d2, 0.0, 1.0)*T*np.exp(-r*T)
        volga_g = vega_g*d1*d2/sigma

    return [price, delta_g, gamma_g, vega_g, rho_g, volga_g]


if __name__ == '__main__':



    S = 100
    K = 100
    r = 0.01
    q = 0.0
    T = 2.0
    sigma = 0.15

    n_step = 100
    s_range = np.linspace(10, 200, n_step + 1)


    call_price_range = []
    hedged_call_price_range = []
    delta_hedged_call_price_range = []
    delta_call_price_range = []

    vol_ref = 0.15

    legend_list = []
    s_range_n  = []

    call_price_range = []
    for i in range(1, len(s_range)):

        s_tmp_n = s_range[i]

        s_range_n.append(s_tmp_n)

        out_call_n = euro_vanilla_dividend(s_tmp_n, K, T, r, q, vol_ref, option='call')
        out_call_ref = euro_vanilla_dividend(100, K, T, r, q, vol_ref, option='call')

        call_price_val_n_tmp = out_call_n[0]

        delta_call         = out_call_ref[1]
        gamma_call         = out_call_ref[2]

        delta_s = s_tmp_n - K
        hedged_n_tmp = call_price_val_n_tmp - delta_s*delta_call #- 1/2*delta_s*delta_s*gamma_call


        call_price_range.append(call_price_val_n_tmp)
        hedged_call_price_range.append(hedged_n_tmp)



    plt.plot(s_range_n, call_price_range, '-')
    plt.plot(s_range_n, hedged_call_price_range, 'r-')

    plt.legend(legend_list)
    plt.xlabel('Stock level', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.title('No hedged vs Delta-Gamma hedged Call price', fontsize=14)
    legend_list = ['No hedged', 'Delta hedged']
    plt.legend(legend_list)

    plt.show()

