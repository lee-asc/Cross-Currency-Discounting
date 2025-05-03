import sys
import numpy as np
from scipy.stats import norm
import pandas as pd

def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()








#def zc_price(T, k, theta, sigma, r0):

    #    a_val = A_vsck(k, theta, sigma, T)
    #    b_val = B_vsck(k, 0, T)

    #    p_val = a_val*np.exp(-b_val*r0)

#    return p_val


#def p_option(k, theta, sigma, r0, t, T):

    #    t = 5
    #    T = 7
    #p_t = zc_price(t, k, theta, sigma, r0)
    #p_T = zc_price(T, k, theta, sigma, r0)


    #gamma = (1.0 - np.exp(-2.0*k*(t)))/(2*k)
    #sigma_p = sigma*np.sqrt(gamma)*B_vsck(k, t, T)
    #h = 1.0/sigma_p*np.log(p_T/(strike*p_t)) + sigma_p/2.0

    #price_ = p_T*norm.cdf(h) - p_t*strike*norm.cdf(h-sigma_p)

    #return price_




def LMM_model(time_step,maturity,zero_curve,forward_vols,N):

    forward_rate_trj = []
    steps=int(maturity/time_step)+1
    t=np.zeros(steps)
    time=0
    n_fwd_vols=forward_vols.shape[0]

    for i in range(steps):
        t[i]=time
        time+=time_step
    Delta=np.full((steps-1),time_step)
    Pt=np.zeros(steps)
    for i in range(steps):
        Pt[i]=1/(1+zero_curve[i])**(i+1)
    forward_rate_from_zero=np.zeros((steps-1,steps-1))

    for i in range(steps-1):
        forward_rate_from_zero[i][0]=1/Delta[i]*(Pt[i]/Pt[i+1]-1)

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
                    for q in range(n_fwd_vols):
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


def compute_discount(n_trj, n_y, forward_rate_trj):

    p_trj = []
    for i in range(0, n_trj):
        p_old = 1
        for j in range(0, n_y):
            fwd_tmp = forward_rate_trj[i][j][j]

            p_new = p_old/(1 + fwd_tmp)
            p_old = p_new

        p_trj.append(p_new)

    return p_trj


def compute_zc_price(n_sim, n_y_mat, forward_rate_trj):

    p_mat_trj = []

    for i in range(0, n_trj):
        p_old = 1.0
        for j in range(0, n_y_mat):
            fwd_tmp = forward_rate_trj[i][n_sim + j][n_sim]
            p_new = p_old / (1 + fwd_tmp)
            p_old = p_new
        p_mat_trj.append(p_new)

    return p_mat_trj



if __name__ == '__main__':

    #from matplotlib import plot as plt
    import matplotlib.pyplot as plt

    #forward rates for the next 10Y
    #zero curve vector of 11 elements(maturities 1 year to 11 years)

    delta_time_step = 1.0
    maturity        = 10
    n_trj           = 200

    zero_curve   = np.array([0.0074, 0.0074, 0.0077, 0.0082, 0.0088, 0.0094, 0.0101, 0.0108, 0.0116, 0.0123, 0.0131])
    forward_vols = np.array([[0.1365, 0.1928, 0.1672, 0.1698, 0.1485, 0.1395, 0.1261, 0.1290, 0.1197, 0.1097],
                             [-0.0662, -0.0702, -0.0406, -0.0206, 0, 0.0169, 0.0306, 0.0470, 0.0581, 0.0666],
                             [0.0319, 0.0225, 0, -0.0198, -0.0347, -0.0163, 0, 0.0151, 0.0280, 0.0384]])


    avg_forward_rate_mc, forward_rate_trj = LMM_model(delta_time_step,maturity,zero_curve,forward_vols,n_trj)

    fwd_sim_all = []
    for j in range(0, maturity):
        fwd_sim_trj = []
        for i in range(0, n_trj):
            fwd_ = forward_rate_trj[i][j][j]
            fwd_sim_trj.append(fwd_)
        fwd_sim_all.append(fwd_sim_trj)
    plt.plot(fwd_sim_all)
    plt.xlabel('Simulation time')
    plt.ylabel('Level of interst rate')
    plt.show()
    #FQ(998)
    #print('forward_rate_trj: ', forward_rate_trj[3][1][1])
    #FQ(99)
    #df = pd.DataFrame(avg_forward_rate_mc)
    #df.to_excel(excel_writer="fwd_sim_trj_avg.xlsx")
