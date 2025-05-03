import sys
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



if __name__ == '__main__':

    T = 10
    n_steps = 1000

    dt = T/n_steps

    k1 = 1/1.5
    k2 = 1/5

    r0 = 0.02
    theta = 0.05
    sigma1 = 0.2
    sigma2 = 0.3


    e_list1 =[]; e_list2 =[]
    v_list1 =[]; v_list2 =[]

    t_list = []; e_list = []; v_list = []

    for i in range(0, n_steps):

        t_tmp = dt*i
        e_tmp1 = e_r_vsck(r0, k1, theta, t_tmp)
        e_tmp2 = e_r_vsck(r0, k2, theta, t_tmp)

        v_tmp1 = var_r_vsck(k1, sigma1, t_tmp)
        v_tmp2 = var_r_vsck(k2, sigma2, t_tmp)

        t_list.append(t_tmp)
        e_list1.append(e_tmp1)
        e_list2.append(e_tmp2)

        v_list1.append(v_tmp1)
        v_list2.append(v_tmp2)

    k1 = 'k=%s'%(str(np.round(k1, 2)))
    k2 = 'k=%s'%(str(np.round(k2, 2)))
    s1 = 'sigma1=%s'%(str(np.round(sigma1, 2)))
    s2 = 'sigma2=%s'%(str(np.round(sigma2, 2)))


    plt.plot(t_list, e_list1)
    plt.plot(t_list, e_list2)

    plt.xlabel('Time [year]')
    plt.ylabel('Expected value')
    plt.legend([k1, k2])
    plt.show()

    plt.plot(t_list, v_list1)
    plt.plot(t_list, v_list2)
    plt.xlabel('Time [year]')
    plt.ylabel('Variance')
    plt.legend([s1, s2])
    plt.show()
