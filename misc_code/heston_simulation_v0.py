import sys

import matplotlib.pyplot as plt
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




def simulate_vsck_trj(r0, k, theta, sigma, n_trj, n_time_steps, p_mat):

    r_old = r0*np.ones(n_trj)
    r_int = np.zeros(n_trj)

    dw = np.random.uniform(low=-1.0, high=1.0, size=[n_time_steps,n_trj])

    r_vec = []
    p_vec = []
    t_vec = []
    vsck_discount_trj = []
    vsck_price_trj = []

    for i in range(0, n_time_steps):

        r_new = r_old + k*(theta - r_old)*dt + sigma*dw[i]*np.sqrt(dt)
        r_old = r_new
        p_tmp = zc_price(p_mat, k, theta, sigma, r_new)
        r_int = r_new + r_int
        r_int = r_int * dt
        discount = np.exp(-r_int)

        vsck_discount_trj.append(discount)
        vsck_price_trj.append(p_tmp)

    return vsck_discount_trj, vsck_price_trj


def simulate_lmm_trj(time_step,maturity,zero_curve,forward_vols,N):

    forward_rate_trj = []
    steps=int(maturity/time_step)+1
    t=np.zeros(steps)
    time=0
    p=forward_vols.shape[0]

    for i in range(steps):
        t[i]=time
        time+=time_step
    Delta=np.full((steps-1),time_step)
    B_0=np.zeros(steps)
    for i in range(steps):
        B_0[i]=1/(1+zero_curve[i])**(i+1)
    forward_rate_from_zero=np.zeros((steps-1,steps-1))

    for i in range(steps-1):
        forward_rate_from_zero[i][0]=1/Delta[i]*(B_0[i]/B_0[i+1]-1)

    forward_rate_mc=0
    for n in range(N):
        forward_rate=np.zeros((steps-1,steps-1))
        for i in range(steps-1):
            forward_rate[i][0]=forward_rate_from_zero[i][0]
        for k in range(1,steps-1):
            for j in range(k):
                drift=0
                for i in range(j+1,k+1):
                    sum2=0
                    drift_vol=0
                    risk=0
                    for q in range(p):
                        w=np.random.standard_normal()
                        sum2+=(forward_vols[q][i-j-1]*forward_vols[q][k-j-1])
                        drift_vol+=forward_vols[q][k-j-1]**2
                        risk+=forward_vols[q][k-j-1]*w*np.sqrt(Delta[j])
                    drift+=(Delta[i]*forward_rate[i][j]*sum2)/(1+Delta[i]*forward_rate[i][j])

                forward_rate[k][j+1]=forward_rate[k][j]*np.exp((drift-drift_vol/2)*Delta[j]+risk)

        forward_rate_trj.append(forward_rate)
        forward_rate_mc+=forward_rate
    forward_rate_mc=forward_rate_mc/N

    return forward_rate_mc, forward_rate_trj


def simulate_lmm_trj_(time_step,maturity,zero_curve,forward_vols,N):

    forward_rate_trj = []
    steps=int(maturity/time_step)+1
    t=np.zeros(steps)
    time=0
    p=forward_vols.shape[0]
    for i in range(steps):
        t[i]=time
        time+=time_step
    Delta=np.full((steps-1),time_step)
    B_0=np.zeros(steps)
    for i in range(steps):
        B_0[i]=1/(1+zero_curve[i])**(i+1)
    forward_rate_from_zero=np.zeros((steps-1,steps-1))
    for i in range(steps-1):
        forward_rate_from_zero[i][0]=1/Delta[i]*(B_0[i]/B_0[i+1]-1)
    forward_rate_mc=0
    for n in range(N):
        forward_rate=np.zeros((steps-1,steps-1))
        for i in range(steps-1):
            forward_rate[i][0]=forward_rate_from_zero[i][0]
        for k in range(1,steps-1):
            for j in range(k):
                sum1=0
                for i in range(j+1,k+1):
                    sum2=0
                    sum3=0
                    sum4=0
                    for q in range(p):
                        e=np.random.standard_normal()
                        sum2+=(forward_vols[q][i-j-1]*forward_vols[q][k-j-1])
                        sum3+=forward_vols[q][k-j-1]**2
                        sum4+=forward_vols[q][k-j-1]*e*np.sqrt(Delta[j])
                    sum1+=(Delta[i]*forward_rate[i][j]*sum2)/(1+Delta[i]*forward_rate[i][j])
                forward_rate[k][j+1]=forward_rate[k][j]*np.exp((sum1-sum3/2)*Delta[j]+sum4)

        forward_rate_trj.append(forward_rate)
        forward_rate_mc+=forward_rate
    forward_rate_mc=forward_rate_mc/N
    return forward_rate_mc, forward_rate_trj


def compute_lmm_zc_price(step_y_sim, n_y_mat, forward_rate_trj):

    p_mat_trj = []

    for i in range(0, n_trj):
        p_old = 1.0
        for j in range(0, n_y_mat):
            fwd_tmp = forward_rate_trj[i][step_y_sim + j][n_y_mat]
            p_new = p_old / (1 + fwd_tmp)
            p_old = p_new
        p_mat_trj.append(p_new)

    p_mat_trj = np.asarray(p_mat_trj)

    return p_mat_trj


def compute_lmm_discount(n_trj, n_y, forward_rate_trj):

    p_trj = []
    for i in range(0, n_trj):
        p_old = 1
        for j in range(0, n_y):
            fwd_tmp = forward_rate_trj[i][j][j]

            p_new = p_old/(1 + fwd_tmp)
            p_old = p_new

        p_trj.append(p_new)

    p_trj = np.asarray(p_trj)
    return p_trj


# implementation of MC
import numpy as np
def MCHeston(St, K, r, T, sigma, kappa, theta, volvol, rho,  stdevs, n_trj, timeStepsPerYear):

    timesteps = T * timeStepsPerYear
    dt = 1/timeStepsPerYear    # Define the containers to hold values of St and Vt
    S_t = np.zeros((timesteps, n_trj))
    V_t = np.zeros((timesteps, n_trj))    # Assign first value of all Vt to sigma

    V_t[0,:] = sigma
    S_t[0, :] = St    # Use Cholesky decomposition to
    means = [0,0]
    covs = [[stdevs[0]**2          , stdevs[0]*stdevs[1]*rho],
            [stdevs[0]*stdevs[1]*rho,          stdevs[1]**2]]

    Z = np.random.multivariate_normal(means, covs, (n_trj, timesteps)).T
    Z1 = Z[0]
    Z2 = Z[1]

    for i in range(1, timesteps):
        # Use Z2 to calculate Vt
        V_t[i,:] = np.maximum(V_t[i-1,:] + kappa * (theta - V_t[i-1,:])* dt + volvol*np.sqrt(V_t[i-1,:] * dt) * Z2[i,:],0)
        
        # Use all V_t calculated to find the value of S_t
        S_t[i,:] = S_t[i-1,:] + r * S_t[i,:] * dt + np.sqrt(V_t[i,:] * dt) * S_t[i-1,:] * Z1[i,:]    


    call_price = np.mean(S_t[timesteps-1, :]- K)
    return S_t, call_price


if __name__ == '__main__':

    St = 100
    K = 120
    r = 0.05
    T = 3
    sigma = 0.3
    kappa = 0.2
    theta = 0.5
    volvol = 0.2
    rho = 0.5
    n_trj = 1000
    timeStepsPerYear = 12

    n_T = timeStepsPerYear*T - 1
    stdevs = [1.0/3.0, 1.0/3.0]

    S_t, call_price = MCHeston(St, K, r, T, sigma, kappa, theta, volvol, rho, stdevs, n_trj, timeStepsPerYear)


    plt.plot(S_t)
    plt.show()

    plt.hist(S_t[n_T,:], density=True, bins=50)
    plt.show()

