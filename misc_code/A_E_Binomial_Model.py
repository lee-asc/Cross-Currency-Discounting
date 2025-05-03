# --- TUTORIAL 5              -------------------------------------------------
#
#     BINOMIAL TREE                                                                     
# 
# --- Define Libraries          -----------------------------------------------

import math as m
import numpy as np

# --- Define Input Data       -------------------------------------------------

cp = +1     # +1/-1 for Call/Put
ae = -1     # +1/-1 for American/European
S	= 60
K	= 65
r	= 0.08
T	= 0.25
sigma	= 0.3


n = 100

# --- Calculation Other Values   ----------------------------------------------

dt = T / n              # <- h
u = m.exp(sigma * m.sqrt(dt))
d = 1 / u
p = (m.exp(r * dt) - d) / (u - d)    
    
# --- Define Result Matrix    -------------------------------------------------

opt_val = np.zeros((n+1, n+1))
S_val = np.zeros((n+1, n+1))



    
    # Generate the Underlying Stock Price ----------
 
S_val[0,0] = S
for i in range(1,n+1):
    S_val[i,0] = S_val[i-1,0] * u
    for j in range(1,i+1):
        S_val[i,j] = S_val[i-1,j-1] * d
            
  
  #Backward recursion for option price -----
    
for j in range(n+1):
    opt_val[n,j] = max(0, cp*( S_val[n,j]-K )) #payoff at T
    
for i in range(n-1,-1,-1):
    for j in range(i+1):
            
            opt_val[i,j] = ( p * opt_val[i+1,j] + (1-p) * opt_val[i+1,j+1] ) / m.exp(r * dt)
            
            if ae == 1:
                opt_val[i,j] = max( opt_val[i,j], cp * (S_val[i,j] - K) )
            
            
            
if cp == 1:
    Call_or_Put = "Call"
        
else:
    Call_or_Put = "Put"
        
if ae == 1:
    Type= "American"
        
else:
    Type= "European"
       
    
print('The price of the', Type, Call_or_Put, 'option is %.5f.' % opt_val[0,0])
