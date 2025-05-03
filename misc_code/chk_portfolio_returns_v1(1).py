
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    file_data_eu = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\eu_data.xlsx"
    file_data_jp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\jp_data.xlsx"
    file_data_sp = r"C:\Users\proprietario\PycharmProjects\test_tutorial_v0\financial_risk_mgm_40720\data\sp_data.xlsx"

    ts_jp = pd.read_excel(open(file_data_jp, 'rb'))
    ts_eu = pd.read_excel(open(file_data_eu, 'rb'))
    ts_sp = pd.read_excel(open(file_data_sp, 'rb'))

    sp_values = ts_sp['SP_500_IDX']
    eu_values = ts_eu['MSCI_EUROPE']
    jp_values = ts_jp['MSCI_JAPAN']


    ptf_value = 1.0*sp_values + 1.0*jp_values + 1.0*eu_values

    ln = len(ptf_value)

    ln = 100
    return_ptf_all = (ptf_value[ln - 1] - ptf_value[0])/ptf_value[0]

    return_eu = (eu_values[ln - 1] - eu_values[0])/eu_values[0]
    return_sp = (sp_values[ln - 1] - sp_values[0])/sp_values[0]
    return_jp = (jp_values[ln - 1] - jp_values[0])/jp_values[0]

    w_sp = sp_values[0] / ptf_value[0]
    w_jp = jp_values[0] / ptf_value[0]
    w_eu = eu_values[0] / ptf_value[0]

    return_ptf_sum = w_eu*return_eu + w_jp*return_jp + w_sp*return_sp


    print('return_ptf_sum: %.4f'%(return_ptf_sum))
    print('return_ptf_all: %.4f'%(return_ptf_all))

