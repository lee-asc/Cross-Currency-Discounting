# --- TUTORIAL 12              -------------------------------------------------
#       FINITE DIFFERENCE - CRANK-NICOLSON
#           Student:                                                                 
#           .....................     
# --- Define Libraries          -------------------------------------------------
from scipy.interpolate import interp1d 
import numpy as np
from scipy.sparse import diags
from numpy.linalg import inv

# --- Define Input Data       -------------------------------------------------

S	= 50
K	= 100
r	= 0.1
T	= 1
sigma	= 0.3

Smax = 200
Sb = 40

n = 100
m = 50

# --- Calculation Other Values   ----------------------------------------------

dt = T / n              
ds = (Smax - Sb) / m
vec_i = np.linspace(Sb/ds, Smax/ds,m+1)

# --- Define Result Matrix    -------------------------------------------------

opt_val = np.zeros((m+1, n+1))
g = np.zeros((m-1, n))
# --- Main Function          --------------------------------------------------

#     
#    # Set up Coefficients Matrix ----------
#    
a = 0.25 * dt * (sigma**2 * np.linspace(1,m-1,m-1) - r) * np.linspace(1,m-1,m-1)
b = - 0.5 * dt * (sigma**2 * np.linspace(1,m-1,m-1)**2 + r)
c = 0.25 * dt * (sigma**2 * np.linspace(1,m-1,m-1) + r) * np.linspace(1,m-1,m-1)
#    

U1 = diags([-a[1:], 1 - b, -c[:-1]], [-1, 0, 1]).toarray()    
U2 = diags([ a[1:], 1 + b,  c[:-1]], [-1, 0, 1]).toarray()
#    
invU1 = inv(U1)
#
#    
#    
#    # Define Buondaries Conditions and g vector
opt_val[:,-1] = np.maximum((K-ds*vec_i ), 0)
opt_val[0,:] = 0
opt_val[-1,:] = 0

for j in range(n-1,-1,-1): 
    g[0,j]  =  a[0]*(opt_val[0,j+1]+opt_val[0,j])
    g[-1,j] =  c[-1]*(opt_val[-1,j+1]+opt_val[-1,j])
    
    
for j in range(n,0,-1):         
    opt_val[1:-1,j-1]    = np.dot(invU1, np.dot(U2, opt_val[1:-1,j]) + g[:,j-1])
               
price = interp1d(ds*vec_i, opt_val[:,0] ) 
print('The price of the Put option is %.5f.' % price(S))