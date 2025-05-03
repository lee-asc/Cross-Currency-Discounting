# --- TUTORIAL 7             -------------------------------------------------
#       Option Pricing with Monte Carlo      
#           Student:                     
#           .....................     


# --- Define Library          -------------------------------------------------

import math as m
import numpy as np


# --- Define Input Data       -------------------------------------------------

cp = +1     # +1/-1 for Call/Put

S	= 60
dS  = 0.5
K	= 65
r	= 0.08
T	= 0.25
sigma	= 0.3



# nr is the number of simulations
nr = 10000

# --- Calculation Other Values   ----------------------------------------------

nu = r - 0.5 * sigma**2



# --- Define Result Matrix    -------------------------------------------------

FinPayOff_1 = np.zeros(( nr, 1))
FinPayOff_2 = np.zeros(( nr, 1))
S_val = np.zeros((nr, 1))
S_val_dS = np.zeros((nr, 1))
rand = np.random.randn(nr, 1)

# --- Main Function          --------------------------------------------------

    
# Process the Monte Carlo Method ----------
    
  
for i in range(nr):
    S_val[i,0] = S * m.exp(nu*T + sigma * T**0.5 * rand[i,0])
    S_val_dS[i,0] = (S+dS) * m.exp(nu*T + sigma * T**0.5 * rand[i,0])

# Mean of Monte Carlo results for option price -----
    
FinPayOff_1 = np.maximum(cp*(S_val[:,-1] - K), 0)
FinPayOff_2 = np.maximum(cp*(S_val_dS[:,-1] - K), 0)
SampleDiff = np.exp( -r * T ) * (FinPayOff_2-FinPayOff_1)/dS
    
Delta = SampleDiff.mean()
    
print('The Delta of the Option computed with Monte Carlo Method with', nr, 'simulations is %.5f.' % Delta)
