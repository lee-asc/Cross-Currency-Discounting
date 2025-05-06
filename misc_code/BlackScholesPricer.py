
import math as m
from scipy.stats import norm


Call_or_Put = "Call"

S	= 37.0
K	= 39
r	= 0.04
T	= 1.0
sigma	= 0.2
q = 0.0


def CallOptionValue(S, K, r, sigma, T, q):
    d1 = (m.log(S/K) + (r -q + 0.5 * sigma**2) * T)/(sigma * m.sqrt(T))
    d2 = d1 - sigma * m.sqrt(T)
    return S *m.exp(-q*T) * norm.cdf(d1) - K * m.exp(-r*T) * norm.cdf(d2)

def PutOptionValue(S, K, r, sigma, T, q):
	d1 = (m.log(S/K) + (r -q + 0.5 * sigma**2) * T)/(sigma * m.sqrt(T))
	d2 = d1 - sigma * m.sqrt(T)
	return K * m.exp(-r*T) * norm.cdf(-d2) - S * m.exp(-q*T) *norm.cdf(-d1)



if Call_or_Put == "Call":
    score = CallOptionValue(S, K, r, sigma, T, q)
    
    
elif Call_or_Put == "Put":
    score = PutOptionValue(S, K, r, sigma, T, q)
    

else:
    score =  "Declaration not Correct. Retray."
    

print('The price of the European', Call_or_Put, 'option is %.5f.' %score)
