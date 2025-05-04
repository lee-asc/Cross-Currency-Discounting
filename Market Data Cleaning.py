spread = pd.read_csv('EURUSD Basis Spreads.csv')
eur_csa = pd.read_csv('EUR CSA.csv')
estr = pd.read_csv('EUR OIS ESTR.csv')

array1 = pd.DataFrame(estr).to_numpy()
eur_zerorates = array1[:,1] / 100
eur_tenors = array1[:,0]
estr_df = array1[:,2]

t_max = 51

eur_zerorates = np.insert(eur_zerorates, 0, eur_zerorates[0])
eur_tenors = np.insert(eur_tenors,0,0)
estr_df = np.insert(estr_df,0,1)

eur_zerorates = np.insert(eur_zerorates, len(eur_zerorates), eur_zerorates[-1])
eur_tenors = np.insert(eur_tenors,len(eur_tenors),t_max)
estr_df = np.insert(estr_df,len(estr_df),estr_df[-1])