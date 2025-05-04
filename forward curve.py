def forward_rate(rates, time, dt, step = 0.01):

    n = len(rates)
    forwards = np.zeros(int(n - (dt / step)))    
    
    for i in range(len(forwards)):
        
        z2 = rates[i + int(dt / step)]
        t2 = time[i + int(dt / step)]
        
        z1 = rates[i]
        t1 = time[i]        
        
        forwards[i] = (t2 * z2 - t1 * z1) / dt
        
    return forwards