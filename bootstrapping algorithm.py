
def function(x, tenor0, tenor1, spread, forward, cf_dates, csa_bootstrap):
    
    J_k = 4 * (tenor1 - tenor0)
    ind1 = np.where(cf_dates == tenor1)
    ind0 = (np.where(cf_dates == tenor0))[0]
    csa_k0 = csa_bootstrap[-1]
    
    f = x ** J_k
    
    
    for j in range(1, int(J_k)+1):
        #f += 0.25 * (csa_k0 ** (1 - j / J_k)) * (x ** j) * (forward[(ind0 + (j-1))] + spread)
        f += 0.25 * (csa_k0 ** (1 - j / J_k)) * (x ** j) * (forward[(ind0 + (j-1))] + spread)
    
    for i in range(len(csa_bootstrap)):
        f += 0.25 * (forward[i] + spread) * csa_bootstrap[i]
        
    f += -1
    
    return f
                     
def dfunction(x, tenor0, tenor1, spread, forward, cf_dates, csa_bootstrap):
                     
    J_k = 4 * (tenor1 - tenor0)
    ind1 = np.where(cf_dates == tenor1)
    ind0 = (np.where(cf_dates == tenor0))[0]
    
    csa_k0 = csa_bootstrap[-1]
    
    df = J_k * x ** (J_k - 1)
    
    for j in range(1, int(J_k)+1):
        df += 0.25 * j * (csa_k0 ** (1 - j/J_k)) * (x ** (j-1)) * (forward[(ind0 + (j-1))]+ spread)
    
    return df      