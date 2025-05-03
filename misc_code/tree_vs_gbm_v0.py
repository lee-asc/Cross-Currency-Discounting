import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


# Calculates price of European call via Black-Scholes Formula
def EuroCall(S0, K, T, r, sigma):

  d1 = (np.log(S0/K) + T * (r + (sigma ** 2)/2))/(sigma * np.sqrt(T))
  d2 = (np.log(S0/K) + T * (r - (sigma ** 2)/2))/(sigma * np.sqrt(T))
  value = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

  return value


def multy_period_binomial_tree(S, K, T, r, sigma, N, Option_type):


    u = math.exp(sigma * math.sqrt(T / N)); #( 1 + sigma * sqrt(T / N))
    d = math.exp(-sigma * math.sqrt(T / N));#( 1 - sigma * sqrt(T / N))

    pu = ((math.exp(r * T / N)) - d) / (u - d);
    pd = 1 - pu;
    disc = math.exp(-r * T / N);

    St = [0] * (N + 1)
    C = [0] * (N + 1)

    St[0] = S * d ** N;

    for j in range(1, N + 1):
        St[j] = St[j - 1] * u / d;

    for j in range(1, N + 1):
        if Option_type == 'P':
            C[j] = max(K - St[j], 0);
        elif Option_type == 'C':
            C[j] = max(St[j] - K, 0);

    for i in range(N, 0, -1):
        for j in range(0, i):
            C[j] = disc * (pu * C[j + 1] + pd * C[j]);

    return C[0]
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #ptf_pca_v0.py



    S = 100.0
    K = 110.0
    #T = 2.22
    T = 2.0
    r = 1.0/100.0
    sigma = 30.0/100.0
    N_max = 5000
    N_min = 50
    DN    = 100


    runs1 = list(range(N_min, N_max, DN))
    #runs1 = list(range(3, 4, 1))

    price = []
    bs_price = []

    for i in runs1:
        price.append(multy_period_binomial_tree(S, K, T, r, sigma, i, 'C'))
        bs_price.append(EuroCall(S, K, T, r, sigma))

    plt.plot(runs1, price, label='Binomial Tree')
    plt.plot(runs1, bs_price, label='BS Level')
    plt.xlabel('N. of steps')
    plt.ylabel('Call Option Price level')

    plt.legend(loc='upper right')
    plt.show()
