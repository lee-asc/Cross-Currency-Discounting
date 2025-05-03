
from numpy import *
import matplotlib.pyplot as plt
import sys

random.seed(1)




import numpy as np

from scipy.stats import gamma



def SVG_simulation(n_sim, n_time, dt, theta, nu, sigma, mu, S0):

    #S = S0 * cumprod ([ ones (1 , T ); exp ( mu * dt + theta * g + sigma * sqrt ( g ).* randn ( N_Sim , T ))]);

    g = gamma.rvs(dt/nu, scale=nu, size=[n_sim, n_time])
    z = np.random.normal(mu, sigma, size = (n_sim, n_time))
    A =  mu*dt + theta*g + sigma*sqrt(g)*z
    X = np.exp(A)
    X0 = np.ones((n_sim,1))
    X2 = np.concatenate((X0, X), axis=1)
    S = S0 * cumprod(X2, 1)

    return S


if __name__ == '__main__':

    a = 1
    nu = 0.1

    n_sim = 1000
    dt = 0.5
    S0 = 100
    n_time =12
    theta = 0
    mu = 0

    sigma = 0.3

    S_t = SVG_simulation(n_sim, n_time, dt, theta, nu, sigma, mu, S0)

    for i in range(0,n_sim):
        plt.plot(S_t[i,])
    plt.show()

    print(S_t.shape)
    plt.hist(S_t[:,n_time], density=True, bins=50)
    plt.axis()
    plt.show()

