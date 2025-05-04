
import timeit

for i in range(3, len(tenors)-1):
#for i in range(3, 12):

    J_k = 4 * (tenors[i] - tenors[i-1])

    root, iterations = newton_raphson(function, dfunction, csa_bootstrap[-1]**(1/J_k), tenors[i], tenors[i+1], 
                                      xccy_bs[i+1], f_eur, cf_dates, csa_bootstrap)

    J_k = 4 * (tenors[i+1] - tenors[i])

    
    print(f"CSA_{tenors[i+1]}: {root ** J_k}, found in {iterations} iterations.")
    
    J_k = 4 * (tenors[i] - tenors[i-1])

    print("NS-Solution Profile:")
    print(timeit.timeit(lambda: newton_raphson(function, dfunction, csa_bootstrap[-1]**(1/J_k), tenors[i], tenors[i+1], 
                                      xccy_bs[i+1], f_eur, cf_dates, csa_bootstrap), number=10000))
    J_k = 4 * (tenors[i+1] - tenors[i])

    csa_tenor1 = root ** J_k

    print(f'At Tenor = {tenors[i+1]}')
    print('diff CSA_bootstrapped v BB csa in bps', (csa_tenor1 - bb_csa[i+1])/0.0001)
    
    
    quarters = np.array((csa_quaters(csa_tenor1, csa_bootstrap, tenors[i], tenors[i+1]))).astype(float)
    
    
    print("Extrapolate Quarters Profile:")
    print(timeit.timeit(lambda: csa_quaters(csa_tenor1, csa_bootstrap, tenors[i], tenors[i+1]), number=10000))
    
    csa_bootstrap = np.concatenate((csa_bootstrap, quarters), axis=None)
    csa_bootstrap = np.concatenate((csa_bootstrap, csa_tenor1), axis=None)

    print('--------------------------------------------------------------------------------')