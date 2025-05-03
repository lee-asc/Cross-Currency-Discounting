# --- TUTORIAL 7             -------------------------------------------------
#       DELTA OPTION CALL WITH MONTE CARLO       
#           Student:                     
#.....................     
# --- Define Library          -------------------------------------------------

import math as m
from scipy.stats import norm
import numpy as np

# --- Define Input Data       -------------------------------------------------


S	= 60
K	= 65
r	= 0.08
T	= 0.25
sigma	= 0.3
q = 0.0

nr = 10000

# --- Calculation Other Values   ----------------------------------------------

nu = r - 0.5 * sigma**2

# --- Define Result Matrix    -------------------------------------------------

Delta = np.zeros((nr, 1))
S_val = np.zeros((nr, 1))
rand = np.random.randn(nr, 1)

# --- Define Indicator Function  -------------------------------------------------
 
def I(x):
    
    if x > 0:
        return 1
    else:
        return 0

# --- Main Function          --------------------------------------------------


for i in range(nr):
    S_val[i,0] = S * m.exp(nu*T + sigma * T**0.5 * rand[i,0])
            
# option Delta results for option price -----
    
for i in range(nr):
    Delta[i,0] = np.exp( -r * T) * (S_val[i,0] / S ) * I(S_val[i,0]-K)
     
mean, std = norm.fit(Delta)
        
print(('The Delta of the Call Option with %i simulations is: Delta = %.5f') % (nr, mean))
print
print(('The standard deviation is %.5f') % (std))

d1 = (m.log(S/K) + (r -q + 0.5 * sigma**2) * T)/(sigma * m.sqrt(T))
DBS = norm.cdf(d1)
print(('The Delta of the option with the BS model is %.5f') % (DBS))