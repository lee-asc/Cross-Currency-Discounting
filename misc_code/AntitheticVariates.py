
import math as m
import numpy as np


cp = -1     # +1/-1 for Call/Put
S	= 60
K	= 65
r	= 0.08
T	= 0.25
sigma	= 0.3
n = 25
nr = 500


nu = r - 0.5 * sigma**2
dt = T / n


FinPayOff = np.zeros(( int(nr/2), 1))
S_val = np.zeros((nr, n+1))
rand = np.random.randn(int(nr/2), n) 


S_val[:,0] = S
for i in range(0, nr, 2):
    for j in range(1,n+1):
        S_val[i,j] = S_val[i,j-1] * m.exp(nu*dt + sigma * dt**0.5 * rand[int(i/2),j-1])
        S_val[i+1,j] = S_val[i+1,j-1] * m.exp(nu*dt - sigma * dt**0.5 * rand[int(i/2),j-1])
            
    
    FinPayOff[int(i/2)] = 0.5*(np.maximum(cp*(S_val[i,-1] - K), 0) + np.maximum(cp*(S_val[i+1,-1] - K), 0))

       
PDisc = np.exp( -r * T ) * FinPayOff
    
mean = PDisc.mean()
std = np.std(PDisc)
    
if cp == +1:
   Call_or_Put = "Call"
        
else:
   Call_or_Put = "Put"
    
print(('The price of the ' + Call_or_Put + ' option with the Antithetic Variates Method is Price: %.5f') % mean)
print(('The standard deviation of the price is: sdt = %.5f') % std)
print(('The number of simulations is  %.5f') % nr)
