# --- TUTORIAL 6        -------------------------------------------------
#       Option Pricing with Monte Carlo Methods  
#           Student:                     
#           .....................     


# --- Define Library          -------------------------------------------------

import math as m
import numpy as np
import matplotlib.pyplot as plt

# --- Define Input Data       -------------------------------------------------

cp = -1     # +1/-1 for Call/Put
S	= 60
K	= 65
r	= 0.08
T	= 0.25
sigma	= 0.3


# n DEFINES THE TIME STEP
n = 25
# nr is the number of simulations
nr = 1000


# --- Calculation Other Values   ----------------------------------------------
# Here we refer to the risk-neutral valuation.
nu = r - 0.5 * sigma**2
dt = T / n              # <- h

# --- Define Result Matrix    -------------------------------------------------

FinPayOff = np.zeros(( nr, 1))
S_val = np.zeros((nr, n+1))
rand = np.random.randn(nr, n)


    
    # Process the Monte Carlo Method ----------
    
S_val[:,0] = S
for i in range(nr):
    for j in range(1,n+1):
        S_val[i,j] = S_val[i,j-1] * m.exp(nu*dt + sigma * dt**0.5 * rand[i,j-1])
            
    # Mean of Monte Carlo results for option price -----
    
FinPayOff = np.maximum(cp*(S_val[:,-1] - K), 0)
       
PDisc = np.exp( -r * T ) * FinPayOff
price = PDisc.mean()
std = np.std(PDisc)

    
# Plot ----
plt.figure()
for i in range(nr):
    plt.plot(S_val[i,:])
plt.title('Monte Carlo Method simulations -  GBM Paths')
    
print('The price of the Monte Carlo Method with', nr, 'simulations is %.5f.' % price)
print('The standard deviation is %.5f.' % std)
