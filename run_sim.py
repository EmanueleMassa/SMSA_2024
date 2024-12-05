import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import scipy.optimize as opt 
from required_functions import gen_model, gauss_model, pwe_model, c_index, bases, get_omega
import time
from joblib import Parallel, delayed
import pandas as pd 

#sample size
n = 400
#covariate to sample size  ratio
zeta = 0.5
#number of covariates 
p = int(n * zeta)
#true signal strength
theta0 = 1.0
#regularization path
etas = np.linspace(6.0, 0.1, 30) #strenght
alpha = 0.01
#ripetitions
m = 10
#simulate the model parameters from the prior
beta0 = rnd.normal(size = p)
beta0 = theta0 * beta0 / np.sqrt(beta0 @ beta0)
#define the true parameters of the cumulative hazard
phi0 = - np.log(2)
rho0 = 2.0
model = 'log-logistic'
#define the interval in which the censoring is uniform
tau1 = 1.0
tau2 = 3.0
#define the knots
n_intervals = 10
tps = np.linspace(0.0, tau2, n_intervals)
#covariance matrix
A0 = np.diag(np.ones(p))
#data generating process
GM = gen_model(A0, beta0, phi0, rho0, tau1, tau2, model) 
#define the fitting model 
pwe = pwe_model(p, tps)
#define the time points where to evaluated the Brier Score 
n_eval = 1000
T_eval = np.linspace(0.0,tau2,n_eval)
delta = tau2/n_eval
b_eval, B_eval = bases(T_eval, tps)
#generate test population
T_test, C_test, X_test = GM.gen(n)
#true survival function
S0 = np.exp(- np.outer(GM.ch(T_eval), np.exp(X_test @ beta0)))
#data containers
r_ibs = np.zeros(len(etas))
def experiment(counter):
    tic = time.time()
    #generate data
    pop = GM.gen(n)
    T, C, X = pop
    S0 = np.exp( - np.outer(GM.ch(T_eval), np.exp(X_test @ GM.beta)))
    #fit the model
    pwe.fit(T, C, X, etas, alpha)
    #null model
    omega_null = get_omega(pwe.F, pwe.B, np.ones(n), alpha)
    H_null = B_eval @ np.exp(omega_null)
    S_null = np.exp(- np.outer(H_null, np.ones(n)))
    ibs_ref = np.sum( (S_null - S0)**2)*delta/n
    #get the betas along the path
    betas = pwe.betas
    #get the S along the path and compute mse_S
    for i in range(len(etas)):
        elp_test = np.exp(X_test @ betas[i,:])
        H_test = B_eval @ np.exp(pwe.omegas[i,:])
        S_test = np.exp(- np.outer(H_test, elp_test))
        ibs_test = np.sum((S_test - S0)**2)*delta/n
        r_ibs[i] = ibs_test/ibs_ref
    #compute w
    w = (betas @ GM.beta)/np.sqrt(GM.beta @ GM.beta)
    #compute v
    v = np.sqrt(np.sum(betas**2, axis = 1) - w**2)
    #compute the test c - index
    c_ind = np.array([c_index(T_test, C_test, X_test @ betas[i,:]) for i in range(len(etas))], float)
    toc = time.time()
    print('experiment '+str(counter)+ ' time elapsed = '+str((toc-tic)/60))
    return w, v, r_ibs, c_ind


# tic = time.time()
# results = Parallel(n_jobs=12)(delayed(experiment)(counter) for counter in range(m))
# t_df = pd.DataFrame(results)
# w = np.stack(t_df.iloc[:, 0].to_numpy())
# v = np.stack(t_df.iloc[:, 1].to_numpy())
# R_ibs = np.stack(t_df.iloc[:, 2].to_numpy())
# c_ind = np.stack(t_df.iloc[:, 3].to_numpy())
# toc = time.time()
# print('total elapsed time = ' + str((toc-tic)/60))

w = np.zeros((m, len(etas)))
v = np.zeros((m, len(etas)))
R_ibs = np.zeros((m, len(etas)))
c_ind = np.zeros((m, len(etas)))

big_tic = time.time()
for i in range(m):
    tic = time.time()
    w[i,:], v[i,:], R_ibs[i,:], c_ind[i,:] = experiment(i)
    toc = time.time()
    print('elapsed time = ' + str((toc-tic)/60))
big_toc = time.time()
print('total elapsed time = ' + str((big_toc-big_tic)/60))


data = {
    'etas' : etas,
    'w_mean' : np.mean(w, axis = 0),
    'w_std' : np.std(w, axis = 0),
    'v_mean' : np.mean(v, axis = 0),
    'v_std' : np.std(v, axis = 0),
    'R_ibs_mean' : np.mean(R_ibs, axis = 0),
    'R_ibs_std' : np.std(R_ibs, axis = 0),
    'c_ind_mean' : np.mean(c_ind, axis = 0),
    'c_ind_std' : np.std(c_ind, axis = 0)

}

df = pd.DataFrame(data)
df.to_csv('data/sim_zeta'+"{:.2f}".format(zeta)+'_alpha'+"{:.2f}".format(alpha)+'.csv', index = False)



