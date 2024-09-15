#%%
import os 
import numpy as np 
import pandas as pd 


def perm_test(xs, ys, nmc):
    
    """_summary_
    
    Permutation test:
    The null hypothesis is that both results
    come from the same distribution 
    
    
    xs =  Results from one experiment
    ys = Results from a different experiment 
    
    nmc = number of permutations  

    Returns:
        int _type_: P-value 
    """
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    list_diff=np.empty(nmc)
    for j in range(nmc):
        np.random.shuffle(zs)
        list_diff[j]=np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))

    ll = len(list_diff[list_diff>diff])
    p_value = (ll+1)/1000
    print("P-value: ", p_value)
    return  p_value 


permutations = 999

###########Oirignal onset [48] and window size [10]################## 
### Between category experiment [results] ####

linear_model_bc = [0.57,0.49,0.52,0.44,0.57,0.56,0.50]
bilstm_model_bc= [0.51,0.56,0.61,0.52,0.54,0.55,0.44]
#bilstm_model_bc= [0.9,0.88,0.78,0.99,0.96,0.92,0.79]

p_value_bc = perm_test(linear_model_bc, bilstm_model_bc,permutations)

###within category experiment  [results] ###

linear_model_wc = [0.55,0.48,0.57,0.51,0.50,0.50,0.45]
bilstm_model_wc=  [0.49,0.58,0.63,0.57,0.57,0.55,0.48]

p_value_wc = perm_test(linear_model_wc, bilstm_model_wc,permutations)

###Leave two out experiment [results]###
linear_model_lo= [0.57,0.47,0.52,0.55,0.55,0.49,0.48]
bilstm_model_lo= [0.40,0.56,0.56,0.56,0.6,0.49,0.44]

p_value_lo = perm_test(linear_model_lo, bilstm_model_lo,permutations)



# %%
import matplotlib.pyplot as plt

Subjects = [1, 2, 3, 4, 5, 6]
bi_results = [0.0542, 0.0539, 0.0545, 0.0559, 0.0556, 0.0553]
linear_results = [0.393,0.374,0.372,0.372,0.372,0.372]

plt.figure(figsize=(10, 6))
plt.plot(Subjects, bi_results, marker='o', linestyle='-', color='b')
plt.plot(Subjects, linear_results, marker='o', linestyle='-', color='r')
plt.xlabel('Subjects',fontsize=16)
plt.ylabel('MSE',fontsize=14)
plt.grid(True)
plt.legend(['Bi-LSTM', 'Linear'])
plt.show()
# %%
