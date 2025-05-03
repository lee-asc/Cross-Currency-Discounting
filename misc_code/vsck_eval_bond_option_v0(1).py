import sys
import numpy as np
from scipy.stats import norm

def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



def e_r_vsck(r0, k, theta, T):

    e_val = r0*np.exp(-k*T)  + theta*(1.0 - np.exp(-k*T))

    return e_val


def var_r_vsck(k, sigma, T):

    v_val = (sigma**2)/(2*k)*(1.0 - np.exp(-2*k*T))

    return v_val


def B_vsck(k, t, T):

    gamma = 1.0 - np.exp(-k*(T-t))
    b_val = 1.0/k*gamma

    return b_val

def A_vsck(k, theta, sigma, T):


    alfa    = theta - sigma*sigma/(2.0*k*k)
    a_val  = np.exp(alfa*(B_vsck(k, 0, T) - T) - sigma*sigma/(4*k)*B_vsck(k,0, T)*B_vsck(k,0, T))

    return a_val


def zc_price(T, k, theta, sigma, r0):

    a_val = A_vsck(k, theta, sigma, T)
    b_val = B_vsck(k, 0, T)

    p_val = a_val*np.exp(-b_val*r0)

    return p_val


def p_option(k, theta, sigma, r0, t, T):

    t = 5
    T = 7
    p_t = zc_price(t, k, theta, sigma, r0)
    p_T = zc_price(T, k, theta, sigma, r0)


    gamma = (1.0 - np.exp(-2.0*k*(t)))/(2*k)
    sigma_p = sigma*np.sqrt(gamma)*B_vsck(k, t, T)
    h = 1.0/sigma_p*np.log(p_T/(strike*p_t)) + sigma_p/2.0

    price_ = p_T*norm.cdf(h) - p_t*strike*norm.cdf(h-sigma_p)

    return price_

if __name__ == '__main__':



    k = 1.0/10
    theta = 0.1
    sigma = 0.5
    r0 = 0.01
    strike = 0.9

    T_sim = 10 #years
    T_exec_opt = 5

    n_time_steps = T_sim*365
    dt = 1.0/365.0

    n_trj = 1000
    r_old = r0*np.ones(n_trj)
    r_int = np.zeros(n_trj)

    dw = np.random.uniform(low=-1.0, high=1.0, size=[n_time_steps,n_trj])


    r_vec = []
    p_vec = []
    t_vec = []

    for i in range(0, n_time_steps):

        t_y_tmp = dt*i
        r_new = r_old + k*(theta - r_old)*dt + sigma*dw[i]*np.sqrt(dt)
        r_old = r_new
        p_tmp = zc_price(2, k, theta, sigma, r_new)
        r_int = r_new + r_int

        if (i == 365*T_exec_opt):

            r_int    = r_int*dt
            discount = np.exp(-r_int)
            payoff   = np.maximum(p_tmp - strike, 0)
            price    =  discount*payoff

            p_mean  = np.mean(p_tmp)
            df_mean = np.mean(discount)
            #print('df_mean: ', df_mean)
            #FQ(999)


        r_vec.append(r_new)
        t_vec.append(t_y_tmp)
        p_vec.append(p_tmp)

    p_mean  = np.mean(price)
    p_error = np.std(price)/np.sqrt(n_trj)

    print('Price: ',  np.round(p_mean, 4))
    print('P error: ', np.round(p_error, 4))


