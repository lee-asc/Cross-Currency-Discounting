
# --- TUTORIAL 11              -------------------------------------------------
#       FINITE DIFFERENCE - IMPLICIT METHOD
#           Student:                                                                 
#.....................     


# --- Define Libraries          -----------------------------------------------

from scipy.interpolate import interp1d 
from scipy.sparse import diags
import numpy as np
from numpy.linalg import inv

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
g = np.zeros((m-1, n))

# --- Main Function          --------------------------------------------------
 
a = - 0.5 * dt * (sigma**2 * np.linspace(1,m-1,m-1) - r) * np.linspace(1,m-1,m-1)
b = 1 + dt * (sigma**2 * np.linspace(1,m-1,m-1)**2 + r)
c = - 0.5 * dt * (sigma**2 * np.linspace(1,m-1,m-1) + r) * np.linspace(1,m-1,m-1) 
        
U = diags([a[1:], b, c[:-1]], [-1, 0, 1]).toarray()      
invU = inv(U)
             
#  Define Boundary Conditions--------------------
   
opt_val[:,-1] = np.maximum((K-ds*np.linspace(0,m,m+1)), 0)
opt_val[0,:] = K * np.exp( - r * dt * (np.linspace(n,0,n+1)))
opt_val[-1,:] = 0
      
#    # Define vector g ---------------------
    
for j in range(n):
    g[0,j]  = +  a[0]*opt_val[0,j]
    g[-1,j] = +  c[-1]*opt_val[-1,j]
    
for j in range(n-1,-1,-1):                          
           
     opt_val[1:-1,j] = np.dot(inv(U), opt_val[1:-1,j+1] - g[:,j] )
            
price = interp1d(ds*np.linspace(0,m,m+1), opt_val[:,0] ) 
print('The price of the Put option is %.5f.' % price(S))