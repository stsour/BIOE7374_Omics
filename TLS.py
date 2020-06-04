import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.linalg as la


# Credit for this function goes to: https://towardsdatascience.com/total-least-squares-in-comparison-with-ols-and-odr-f050ffc1a86a
def tls(X,y):
    if X.ndim == 1:
        n = 1 # the number of variable of X
        X = X.reshape(len(X),1)
    else:
        n = np.array(X).shape[1] 
    
    Z = np.vstack((X.T,y)).T #create new matrix A with x and y 
    U, s, Vt = la.svd(Z, full_matrices=True) # singular value decomposition
    
    V = Vt.T
    Vxy = V[:n,n:]
    Vyy = V[n:,n:]
    a_tls = - Vxy  / Vyy # total least squares soln, i.e. right singular vector of A 
                        #corresponding to the smallest singular value
    
    return(a_tls[0])

PLS_slopes_dict = {}
TLS_slopes_dict = {}

sample_size = [50, 100, 500, 1000]
stdev = [0.1, 0.25, 0.5, 0.75, 1]

for N in sample_size: 
    for S in stdev:
        print(N,S)
        PLS_slopes = []
        TLS_slopes = []
        for i in range(1000):
            x = stats.uniform.rvs(size=N) #randomly sample N from uniform distribution [0,1]
            y = 2*x #calculate y array
            x_noise = stats.norm.rvs(size=N, scale=S) #generate noise in x
            y_noise = stats.norm.rvs(size=N, scale=S) #generate noise in y
            x = np.concatenate((x, x_noise)) #add noise to x
            y = np.concatenate((y, y_noise)) #add noise to y

            #PLS regression on 1D data is just normal linear least squares regression
            pls_A = np.vstack([x, 2*np.ones(len(x))]).T
            pls_slope, yint = la.lstsq(pls_A, y, rcond=None)[0]
            PLS_slopes.append(pls_slope)
            
            #TLS
            tls_slope = tls(x,y)[0]
            TLS_slopes.append(tls_slope)
            
        PLS_slopes_dict[str(N)+'_'+str(S)] = PLS_slopes
        TLS_slopes_dict[str(N)+'_'+str(S)] = TLS_slopes

# generate violin plot for each sample size-standard deviation pair
for i,N in enumerate(sample_size): 
    for j,S in enumerate(stdev):
        fig, ax = plt.subplots(figsize=(5,5))
        pls_slopes = PLS_slopes_dict[str(N)+'_'+str(S)]
        tls_slopes = TLS_slopes_dict[str(N)+'_'+str(S)]
        label_vec = ['PLS']*len(pls_slopes) + ['TLS']*len(tls_slopes)
        plot_df = pd.DataFrame(zip(pls_slopes+tls_slopes, label_vec), columns = ['Slope', 'Method'])
        
        sns.violinplot(x='Method', y='Slope', data=plot_df, cut=0)
        plt.title('N = '+str(N)+', s='+str(S))
        plt.savefig('Results/PLS_vs_TLS_'+str(N)+'_'+str(S)+'.png')
