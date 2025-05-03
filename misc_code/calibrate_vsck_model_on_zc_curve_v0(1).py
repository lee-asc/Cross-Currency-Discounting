import time


from scipy import optimize

import matplotlib.pyplot as plt
from os import sys
import numpy as np
from scipy.stats import norm



#from dateutil.relativedelta import *


from math import exp, log


def FQ(ref):

	print ('----------FIN QUI (%s) TUTTO OK-------'%(ref))
	sys.exit()



def compute_zc_cir_rate(list_model_params, T):


	r0   = list_model_params[0]
	kappa = list_model_params[1]
	theta   = list_model_params[2]
	sigma   = list_model_params[3]
	
	h = (kappa*kappa + 2.0*sigma*sigma)**(0.5)

	g0 = 2*kappa*theta/(sigma*sigma)
	g1 = exp(T*h) - 1.0
	g2 = exp(T*(h + kappa)/2.0)

	A0 = (2*h*g2/(2.0*h + (kappa + h)*g1))
	B0 = (2.0*g1/(2.0*h + (kappa + h)*g1))
	
	model_rate = -(g0*log(A0) - B0*r0)/T
	

	return model_rate


def zc_price(T, k, theta, sigma, r0):

    a_val = A_vsck(k, theta, sigma, T)
    b_val = B_vsck(k, 0, T)

    p_val = a_val*np.exp(-b_val*r0)

    return p_val

def zc_call_bond_option(k, theta, sigma, r0, strike, t, T):

    p_t = zc_price(t, k, theta, sigma, r0)
    p_T = zc_price(T, k, theta, sigma, r0)

    gamma = (1.0 - np.exp(-2.0*k*(t)))/(2*k)
    sigma_p = sigma*np.sqrt(gamma)*B_vsck(k, t, T)
    h = 1.0/sigma_p*np.log(p_T/(strike*p_t)) + sigma_p/2.0

    price_ = p_T*norm.cdf(h) - p_t*strike*norm.cdf(h-sigma_p)

    return price_


def zc_put_bond_option(list_model_params, strike, t, T):

	r0 = list_model_params[0]
	k	  = list_model_params[1]
	theta = list_model_params[2]
	sigma = list_model_params[3]


	p_t = zc_price(t, k, theta, sigma, r0)
	p_T = zc_price(T, k, theta, sigma, r0)
	gamma = (1.0 - np.exp(-2.0*k*(t)))/(2*k)
	sigma_p = sigma*np.sqrt(gamma)*B_vsck(k, t, T)
	h = (1.0/sigma_p)*np.log(p_T/(strike*p_t)) + sigma_p/2.0
	price_ = -(p_T*norm.cdf(-h) - p_t*strike*norm.cdf(-(h-sigma_p)))
	#price_ = +(p_T*norm.cdf(+h) - p_t*strike*norm.cdf(+(h-sigma_p)))

	#price_ = 2*price_
	#price_ = + p_t*strike*norm.cdf(-h+sigma_p)

	return price_


def B_vsck(k, t, T):

    gamma = 1.0 - np.exp(-k*(T-t))
    b_val = 1.0/k*gamma

    return b_val

def A_vsck(k, theta, sigma, T):


    alfa    = theta - sigma*sigma/(2.0*k*k)
    a_val  = np.exp(alfa*(B_vsck(k, 0, T) - T) - sigma*sigma/(4*k)*B_vsck(k,0, T)*B_vsck(k,0, T))

    return a_val

	
def compute_zc_vsck_rate(list_model_params, T):


	r0      = list_model_params[0]
	kappa   = list_model_params[1]
	theta   = list_model_params[2]
	sigma   = list_model_params[3]

	B0 = (1.0/kappa)*(1.0 - exp(-kappa*T))
	g0 = (sigma*sigma)/(4.0*kappa)
	G0 = (theta - sigma*sigma/(2.0*kappa*kappa))
	A0 = exp(G0*(B0 - T) - g0*B0*B0)

	model_rate = -(log(A0) - B0*r0)/T
	
	return model_rate

	
	



def loss_zc_model_vsck(list_model_params, prices_dict):


	time_mkt_list = list(prices_dict.keys())
	time_mkt_list.sort()
	
	diff_sum = 0.0

	for time_tmp in time_mkt_list:

		time_tmp = float(time_tmp)
		model_price_tmp  = compute_zc_vsck_rate(list_model_params, time_tmp)
		mkt_price_tmp    = prices_dict[time_tmp]
		diff = abs(model_price_tmp - mkt_price_tmp)
		diff = diff*diff
		diff_sum = diff_sum +  diff

	return diff_sum


def loss_caplet_model_vsck(list_model_params, prices_dict, tau):
	diff_sum = 0.0

	time_mkt_list = list(prices_dict.keys())
	time_mkt_list.sort()


	for time_tmp in time_mkt_list:
		time_tmp = float(time_tmp)

		#model_price_tmp = zc_put_bond_option(list_model_params, strike, time_tmp - tau, time_tmp)
		#model_price_tmp = 100.0*(1.0 + strike*tau)*model_price_tmp
		#model_price_tmp = compute_zc_vsck_rate(list_model_params, time_tmp)

		model_price_tmp = caplet_price(list_model_params, time_tmp, tau)
		#print('model_price_tmp: ', model_price_tmp)
		mkt_price_tmp = prices_dict[time_tmp]
		diff = abs(model_price_tmp - mkt_price_tmp)
		diff = diff * diff
		diff_sum = diff_sum + diff

	return diff_sum


def load_curve_fromFile(inputFile, c_type):

	fin = open(inputFile, 'r')
	listInput = fin.readlines()
	n_lines = len(listInput)
			
	
	py_values = []
	py_times = []
	py_dict = {}

	
	for i in range(1, n_lines):
		
		line_splitted = listInput[i].split(";")
				
		timeTmp     = float((line_splitted[0]))	
		py_valTmp   = float(line_splitted[1])
		
		if (timeTmp > (1.0/365.2425)):

			if (c_type == 'SMP'):

				py_valTmp = -1.0/timeTmp*log(1.0/(1.0 + py_valTmp*timeTmp))

			elif (c_type == 'CMP'):

				py_valTmp = log(1.0 + py_valTmp)

			else:

				py_valTmp = py_valTmp
			
			py_times.append(timeTmp)
			py_values.append(py_valTmp)
			py_dict[timeTmp] = py_valTmp

		else:
		
			pass
			
	return py_dict


def computeCHI2(mkt_list, mdl_opt_list):


	x2TmpSum = 0.0
	for i in range(0, len(mkt_list)):
	
		mdlTmp = mdl_opt_list[i]
		mktTmp = mkt_list[i]
		
		x2Tmp =  abs(float(mktTmp) - float(mdlTmp))
		
		x2Tmp = x2Tmp/float(mdlTmp)
		
		x2TmpSum = x2TmpSum + x2Tmp 
	
	
	x2TmpSum = float(x2TmpSum/float(i))
	return x2TmpSum
	
def convertRate(regime_in, regime_out, time_in, rate_in):

	if (regime_in == 'CON'):
	
		if (regime_out == 'CMP'):				
			rate_out = (exp(rate_in) - 1.0)
		elif (regime_out == 'SMPL'):
			rate_out = 1.0/time_in*(exp(rate_in*time_in) - 1.0)
		elif (regime_out == 'CON'):
			rate_out = rate_in
		else:
			print ('Regime OUTPUT non gestito!!!!')
		
	else:
			print ('Regime INPUT non gestito!!!!')
	
	return rate_out

def caplet_price(list_model_params_opt, tt, tau):

	strike = 1
	#tau = 0.5
	strike = 1.0
	#strike = 1.0/(1.0 + strike*tau)

	mdl_tmp = zc_put_bond_option(list_model_params_opt, strike, tt - tau, tt)
	mdl_tmp = 10000.0 * (1.0 + strike * tau) * mdl_tmp
	mdl_tmp = mdl_tmp

	return mdl_tmp

if __name__ == "__main__":


	compounding_type = 'CON'
	inputFile = 'input_curve/test_zc_rates_0.csv'

	t_ref = 3.0/12.0
	
	#-------------------------------------------------------

	mkt_prices_dict   = load_curve_fromFile(inputFile, compounding_type)

	r0_0 	= 0.01
	kappa_0 = 0.01
	theta_0 = 1.01
	sigma_0 = 0.01

	r0_min     = 0.01
	kappa_min  = 0.000001
	theta_min  = 0.00001
	sigma_min  = 0.0001

	r0_max 		= 10.0
	kappa_max 	= 10.0
	theta_max 	= 10.0
	sigma_max  	= 10.0

	# ------------ INPUT CURVE -----------------

	
	x0_vsck     = [r0_0, kappa_0, theta_0, sigma_0]
	x_vsck_bnd  = [[r0_min, r0_max], [kappa_min, kappa_max], [theta_min, theta_max], [sigma_min, sigma_max]]
	ff      	= optimize.minimize(loss_zc_model_vsck, x0_vsck, mkt_prices_dict, method = 'TNC',  bounds = x_vsck_bnd)


	list_model_params_opt = []


	r0_opt    = ff.x[0];
	kappa_opt = ff.x[1];
	theta_opt = ff.x[2];
	sigma_opt = ff.x[3];

	print('r0_opt: ', r0_opt)
	print('kappa_opt: ', kappa_opt)
	print('theta_opt: ', theta_opt)
	print('sigma_opt: ', sigma_opt)

	bnd_prms = x_vsck_bnd

	list_model_params_opt.append(r0_opt)
	list_model_params_opt.append(kappa_opt)
	list_model_params_opt.append(theta_opt)
	list_model_params_opt.append(sigma_opt)

	t_mkt_list = list(mkt_prices_dict.keys())

	mdl_opt_list = []
	t_mkt_list_n = []
	mkt_list 	 = []

	t_mkt_list.sort()

	for tt in t_mkt_list:

		mdl_tmp = compute_zc_cir_rate(list_model_params_opt, tt)
		mdl_opt_list.append(float(mdl_tmp))
		mkt_list.append(mkt_prices_dict[tt])

		t_mkt_list_n.append(tt)
		
	t_ref_model = []
	mdl_opt_ref = []
	

	for i in range(0, len(t_mkt_list_n)):
	
		t_ref_model.append(t_mkt_list_n[i])
		mdl_opt_ref.append(mdl_opt_list[i])
	
	

	plt.figure(1)
	plt.plot(t_mkt_list_n, mkt_list, 'o', t_ref_model, mdl_opt_ref, '--')
	plt.title('Fittig results')
	plt.ylabel('ZC rates')
	plt.xlabel('Maturity')
	plt.legend(['MKT', 'Model'])
	plt.show()


	#plt.figure(2)
	#plt.plot(t_mkt_list_n, mkt_list, 'o', t_mkt_list_n, mkt_list, '--')
	#plt.title('Fittig results')
	#plt.ylabel('ZC rates')
	#plt.xlabel('Maturity')
	#plt.legend(['MKT', 'Model'])
	#plt.show()
