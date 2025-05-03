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




def simulate_vsck_trj(r0, k, theta, sigma, n_trj, n_time_steps):

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
        p_tmp = zc_price(2, k, theta, sigma, r_new)
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



def compute_lmm_zc_price(step_y_sim, n_y_mat, forward_rate_trj):

    p_mat_trj = []
    n_trj = len(forward_rate_trj)

    for i in range(0, n_trj):
        p_old = 1.0
        for j in range(0, n_y_mat):
            fwd_tmp = forward_rate_trj[i][step_y_sim + j][n_y_mat]
            p_new = p_old / (1 + fwd_tmp)
            p_old = p_new
        p_mat_trj.append(p_new)

    p_mat_trj = np.asarray(p_mat_trj)

    return p_mat_trj


def compute_lmm_sr_price(n_t, n_mat, dt_sr, forward_rate_trj):

    p0 = compute_lmm_discount(n_t, forward_rate_trj)
    p1 = compute_lmm_discount(n_t + n_mat, forward_rate_trj)

    n_pay = n_mat

    annuity = 0
    for i in range(1, n_pay):
        p_tmp = compute_lmm_discount(i, forward_rate_trj)
        annuity = annuity + dt_sr*p_tmp


    sr_out = (p0 - p1)/annuity


    return sr_out, annuity


def compute_lmm_discount(n_y, forward_rate_trj):

    p_trj = []
    n_trj = len(forward_rate_trj)
    for i in range(0, n_trj):
        p_old = 1
        for j in range(0, n_y):
            fwd_tmp = forward_rate_trj[i][j][j]

            p_new = p_old/(1 + fwd_tmp)
            p_old = p_new

        p_trj.append(p_new)

    p_trj = np.asarray(p_trj)
    return p_trj



if __name__ == '__main__':



    strike = 0.9
    T_sim = 10 #years
    n_time_steps = T_sim*365
    dt = 1.0/365.0
    n_trj = 1000

    delta_time_yy = 1
    maturity = 10

    zero_curve   = np.array([0.0074, 0.0074, 0.0077, 0.0082, 0.0088, 0.0094, 0.0101, 0.0108, 0.0116, 0.0123, 0.0131])
    forward_vols = np.array([[0.1365, 0.1928, 0.1672, 0.1698, 0.1485, 0.1395, 0.1261, 0.1290, 0.1197, 0.1097],[-0.0662, -0.0702, -0.0406, -0.0206, 0, 0.0169, 0.0306, 0.0470, 0.0581, 0.0666],[0.0319, 0.0225, 0, -0.0198, -0.0347, -0.0163, 0, 0.0151, 0.0280, 0.0384]])

    avg_forward_rate_mc, forward_rate_trj    = simulate_lmm_trj(delta_time_yy, maturity, zero_curve, forward_vols,n_trj)


    n_expiry   = 5
    n_swap_mat = 3
    dt_pay     = 1
    strike  = 0.002

    discount_lmm    = compute_lmm_discount(n_expiry, forward_rate_trj)
    sr, annuity     = compute_lmm_sr_price(n_expiry, n_swap_mat, dt_pay, forward_rate_trj)
    swaption_price  = discount_lmm*annuity*np.maximum(sr - strike, 0)

    sr_mean  = np.mean(sr)

    p_mean  = np.mean(swaption_price)
    p_error = np.std(swaption_price)/np.sqrt(n_trj)


    print('p_mean: %.6f'%(p_mean))
    print('p_error: %.6f'%(p_error))

    FQ(77)
