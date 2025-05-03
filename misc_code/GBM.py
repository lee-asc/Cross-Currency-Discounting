# --- TUTORIAL 6              -------------------------------------------------
#       GEOMETRIC BROWNIAN MOTION     
#           Student:                     
#------------------------------------------------------------------------------     


# --- Define Library          -------------------------------------------------

import math as m
import numpy as np
import matplotlib.pyplot as plt

# --- Define Input Data       -------------------------------------------------


S	= 60
K	= 65
r	= 0.08
T	= 0.25
sigma	= 0.3

# n DEFINES THE TIME STEP
n = 100
# nr is the number of simulations
nr = 1000

# --- Calculation Other Values   ----------------------------------------------

# Here we refer to the risk-neutral valuation.
nu = r - 0.5 * sigma**2
dt = T / n              # <- h

# --- Define Result Matrix    -------------------------------------------------


S_val = np.zeros((nr, n+1))
rand = np.random.randn(nr, n)


    
 # Process the Monte Carlo Method ----------
    
S_val[:,0] = S
for i in range(nr):
   for j in range(1,n+1):
      S_val[i,j] = S_val[i,j-1] * m.exp(nu*dt + sigma * dt**0.5 * rand[i,j-1])

    
# Plot ----
plt.figure()
for i in range(nr):
    plt.plot(S_val[i,:])
plt.title('Monte Carlo Method simulations -  GBM Paths')
    