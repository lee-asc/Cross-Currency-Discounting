import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



def e_r_vsck(r0, k, theta, T):

    e_val = r0*np.exp(-k*T)  + theta*(1.0 - np.exp(-k*T))

    return e_val


def var_r_vsck(k, sigma, T):

    v_val = (sigma**2)/(2*k)*(1.0 - np.exp(-2*k*T))

    return v_val


def B_vsck(k, T):

    gamma = 1.0 - np.exp(-k*T)
    b_val = 1.0/k*gamma

    return b_val

def A_vsck(k, theta, sigma, T):


    alfa    = theta - sigma*sigma/(2.0*k*k)
    a_val  = np.exp(alfa*(B_vsck(k, T) - T) - sigma*sigma/(4*k)*B_vsck(k,T)*B_vsck(k,T))

    return a_val


def zc_price(T, k, theta, sigma, r0):

    a_val = A_vsck(k, theta, sigma, T)
    b_val = B_vsck(k, T)

    p_val = a_val*np.exp(-b_val*r0)

    return p_val

if __name__ == '__main__':


    k = 1.0/10
    theta = 0.8
    sigma = 0.1
    r0 = 0.01
    mat_zc = 2.0

    T_sim = 10 #years

    n_time_steps = T_sim*365
    dt = 1.0/365.0

    n_trj = 1000
    r_old = r0*np.ones(n_trj)

    dw = np.random.uniform(low=-1.0, high=1.0, size=[n_time_steps,n_trj])


    r_vec = []
    p_vec = []
    t_vec = []

    for i in range(0, n_time_steps):

        t_y_tmp = dt*i

        r_new = r_old + k*(theta - r_old)*dt + sigma*dw[i]*np.sqrt(dt)
        r_old = r_new

        p_tmp = zc_price(mat_zc, k, theta, sigma, r_new)
        r_vec.append(r_new)
        t_vec.append(t_y_tmp)
        p_vec.append(p_tmp)


    plt.title('Short rate simulation')
    plt.plot(t_vec, r_vec)
    plt.xlabel('Time [year]')
    plt.ylabel('Short rate')
    #plt.legend([k1, k2])

    plt.show()


    plt.title('ZC price simulation')
    plt.plot(t_vec, p_vec)
    plt.xlabel('Time [year]')
    plt.ylabel('Short rate')
    #plt.legend([k1, k2])

    plt.show()

    plt.hist(r_vec[n_time_steps-1], bins = 100)
    plt.xlabel('Interest rate level')
    plt.ylabel('Counts')
    plt.show()