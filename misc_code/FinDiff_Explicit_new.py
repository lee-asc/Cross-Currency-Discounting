# --- TUTORIAL 10              -------------------------------------------------
#       FINITE DIFFERENCES - EXPLICIT METHOD
#           Student:                                                                 
#           .....................     

# --- Define Libraries          -------------------------------------------------

from scipy.interpolate import interp1d 
import numpy as np

# --- Define Input Data       -------------------------------------------------
S	= 62
K	= 50
r	= 0.1
T	= 1
sigma	= 0.3

Smax = 200

n = 100
m = 50

# --- Calculation Other Values   ----------------------------------------------

dt = T / n              
ds = Smax / m
     
# --- Define Result Matrix    -------------------------------------------------
opt_val = np.zeros((m+1, n+1))


# --- Main Function          --------------------------------------------------
    

# Define Boundary Conditions  ---------------------
    
opt_val[:,-1] = np.maximum((K-ds*np.linspace(0,m,m+1)), 0)
opt_val[0,:] = K * np.exp( - r * dt * (np.linspace(n,0,n+1)))
opt_val[-1,:] = 0
      

# Set up Coefficients ----------
    
a = 0.5 * dt * (sigma**2 * np.linspace(1,m-1,m-1) - r) * np.linspace(1,m-1,m-1)
b = 1 - dt * (sigma**2 * np.linspace(1,m-1,m-1)**2 + r)
c = 0.5 * dt * (sigma**2 * np.linspace(1,m-1,m-1) + r) * np.linspace(1,m-1,m-1)
    
    
for j in range(n-1,-1,-1):
   for i in range(1,m):
            
       opt_val[i,j] = a[i-1] * opt_val[i-1,j+1] + b[i-1] * opt_val[i,j+1] + c[i-1] * opt_val[i+1,j+1]
             
price = interp1d(ds*np.linspace(0,m,m+1), opt_val[:,0] ) 
print('The price of the Put option is %.5f.' % price(S))
