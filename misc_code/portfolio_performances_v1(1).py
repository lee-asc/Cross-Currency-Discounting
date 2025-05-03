
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys



# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    file_data_eu = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\eu_data.xlsx"
    file_data_jp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\jp_data.xlsx"
    file_data_sp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\sp_data.xlsx"


    xx_sp = pd.read_excel(open(file_data_sp, 'rb'))
    xx_jp = pd.read_excel(open(file_data_jp, 'rb'))
    xx_eu = pd.read_excel(open(file_data_eu, 'rb'))

    sp_values = xx_sp['SP_500_IDX']
    eu_values = xx_eu['MSCI_EUROPE']
    jp_values = xx_jp['MSCI_JAPAN']

    ln = len(sp_values)

    sp_end   = sp_values[ln - 1]
    sp_start = sp_values[0]

    ref = sp_end/sp_start
    #xx = pd.read_excel(open(file_data, 'rb'), sheet_name='Sheet3')

    ptf_values = 0.2*sp_values + 0.3*jp_values + 0.5*eu_values

    n_bins = 100


    #------ compute log-returns--------------
    #----------------------------------------

    log_return_sp = []
    log_return_eu = []
    log_return_jp = []

    log_return_sum = 0
    return_sum     = 0

    print('ln: ', ln)
    log_r_end = np.log(sp_values[ln-1])        - np.log(sp_values[0])
    r_end     = (sp_values[ln-1]/sp_values[0]) - 1.0

    log_return_ptf = []

    for i in range(0, ln-1):
        log_r_sp_tmp  = np.log(sp_values[i + 1]) - np.log(sp_values[i])
        log_r_jp_tmp  = np.log(jp_values[i + 1]) - np.log(jp_values[i])
        log_r_eu_tmp  = np.log(eu_values[i + 1]) - np.log(eu_values[i])
        log_r_ptf_tmp = np.log(ptf_values[i + 1]) - np.log(ptf_values[i])

        #if (np.abs(log_r_tmp)) < 0.00000001:
        #    continue
        #else:
        #    log_return.append(log_r_tmp)

        log_return_eu.append(log_r_eu_tmp)
        log_return_jp.append(log_r_jp_tmp)
        log_return_sp.append(log_r_eu_tmp)
        log_return_ptf.append(log_r_ptf_tmp)

        #log_return_ptf.append(0.2*log_r_eu_tmp + 0.3*log_r_jp_tmp + 0.5*log_r_sp_tmp)




    print('======== plot level =========')
    plt.plot(eu_values)
    plt.plot(jp_values)
    plt.plot(sp_values)
    plt.plot(ptf_values)
    plt.legend(['MSC_EU', 'MSC_JPY', 'SP500', 'PTF'])


    plt.show()
    #sys.exit()
    print('======== plot return =========')


    plt.plot(log_return_eu)
    plt.plot(log_return_jp)
    plt.plot(log_return_sp)
    plt.plot(log_return_ptf)
    plt.legend(['MSC_EU', 'MSC_JPY', 'SP500', 'PTF'])


    plt.show()
    #sys.exit()

    #plt.plot(log_return_ptf)

    #plt.xlabel('R level ')
    #plt.ylabel('Error: (return - log-return)')

    mean_eu = np.asarray(log_return_eu).mean()
    mean_jp = np.asarray(log_return_jp).mean()
    mean_sp = np.asarray(log_return_sp).mean()
    mean_ptf = np.asarray(log_return_ptf).mean()

    mean_ptf2 = 0.2*mean_eu + 0.3*mean_jp + 0.5*mean_sp

    print('Mean log-ret MSC_EU:  %.6f'%(mean_eu))
    print('Mean log-ret MSC_JP: %.6f'%(mean_jp))
    print('Mean log-ret SP500: %.6f'%(mean_sp))
    print('=======================================')
    print('Mean log-ret ptf: %.6f'%(mean_ptf))
    print('Mean log-ret ptf2: %.6f'%(mean_ptf2))

    plt.show()


    #sys.exit()

    log_return = log_return_eu
    log_return_df = pd.DataFrame(log_return)
    data_to_hist = log_return_df

    # ------ perform histogram ------------
    #--------------------------------------

    ax = data_to_hist.plot.hist(bins=n_bins, alpha=1)
    count, division = np.histogram(data_to_hist, bins=n_bins)

    #plt.show()
    #sys.exit()

    norm_count  = 0.0
    n_mean      = 0.0

    bin_mean = []
    for i in range(0, len(division)-1):
        bin_mean_tmp = division[i] + (division[i+1] - division[i])/2.0
        bin_mean.append(bin_mean_tmp)
        norm_count = norm_count + count[i]

    prob_val = count/norm_count
    n_bins = len(bin_mean)

    #------------------------
    ##-------compute M1 -----
    #------------------------

    m1_x = 0
    #ln =
    #n_bins = ln-1
    for i in range(0, n_bins):

        m1_x  = m1_x + prob_val[i]*bin_mean[i]

    #------------------------
    ##-------compute M2------
    #------------------------
    m2_x = 0
    for i in range(0, n_bins):

        m2_x  = m2_x + prob_val[i]*((bin_mean[i] - m1_x)**2.0)

    sigma_x = np.sqrt(m2_x)
    #print('m1_x: ', m1_x)
    print('=============')
    print('m2_x: ', m2_x)
    print('m1_x: ', m1_x)
    print('sigma_x: ', sigma_x)
    print('n_bins: ', n_bins)
    print('-----------------')
    print('Total: ', np.exp(m1_x*ln))
    #print('Ref: ', ref)
    #print('ln: ', ln)


    #hist_ = sp_values.hist(bins=30)
    plt.show()
    #print('xx: ', hist_)

    #plot
    #delta_ = (x - y)
    #plt.plot(x,delta_)
    #plt.xlabel('R level ')
    #plt.ylabel('Error: (return - log-return)')

    #plt.show()


