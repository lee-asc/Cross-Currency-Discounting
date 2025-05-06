
import math as m
import numpy as np
import matplotlib.pyplot as plt

S	= 60
K	= 65
r	= 0.08
T	= 0.25
sigma	= 0.3

n = 50
nr = 1000


nu = r - 0.5 * sigma**2
dt = T / n

FinPayOff = np.zeros(( nr, 1))
S_val = np.zeros((nr, n+1))
rand = np.random.randn(nr, n)
ControlVars = np.zeros(( nr, 1))

    
S_val[:,0] = S
for i in range(nr):
    for j in range(1,n+1):
        S_val[i,j] = S_val[i,j-1] * m.exp(nu*dt + sigma * dt**0.5 * rand[i,j-1])
            
FinPayOff = np.exp( -r * T ) * np.maximum(np.mean(S_val[:,1:], 1)- K, 0)

    
StockSum = np.sum(S_val,1)
ExpSum = S * ( 1 - np.exp( (n+1) * r * dt )) / ( 1 - np.exp( r * dt ))
    
b = np.cov(StockSum, FinPayOff)[0,1] / np.var(StockSum)
    
for i in range(nr):
    ControlVars[i] = FinPayOff[i] - b * (StockSum[i] - ExpSum)
    
mean = ControlVars.mean()
std = np.std(ControlVars)
    
mean_MC = FinPayOff.mean()
std_MC = np.std(FinPayOff)
    
plt.figure()
for i in range(nr):
    plt.plot(S_val[i,:])
 
    
print('The price of the Asian Option with CV Method and', nr, 'simulations is %.5f.' % mean)
print('The standard deviation of the price with CV Method and', nr, 'simulations is %.5f.' % std)
print('The price of the Option without CV Method and', nr, 'simulations is %.5f.' % mean_MC)
print('The standard deviation of the price withouth CV Method and', nr, 'simulations is %.5f.' % std_MC)
