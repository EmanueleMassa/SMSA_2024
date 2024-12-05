import numpy as np
from required_functions import gauss_model, rs_pwe, c_index, bases
import time
import pandas as pd 

#zeta 
zeta = 2.0
#lambda values 
etas = np.exp(np.linspace(np.log(6.0), np.log(0.1), 100))
alpha = 0.01
#true signal strenght
theta0 = 1.0
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
#population 
m = 2000
GM = gauss_model(theta0, phi0, rho0, tau1, tau2, model) 
pop = GM.data_gen(m)
T, C, Z0, Q = pop
# create the pwe model object 
pwe = rs_pwe(zeta, theta0, tps, pop, GM.ch(T))
#data container
metrics = np.empty((len(etas),6))
#define the time points where to evaluated the Brier Score 
n_eval = 1000
T_eval = np.linspace(0.0,tau2,n_eval)
b_eval, B_eval = bases(T_eval, tps)
#compute the true survival function at T_eval
S0 = np.exp(- np.outer(GM.ch(T_eval),np.exp(theta0*Z0)))
#estimate the model without covariates
omega_null = pwe.get_omega(np.zeros(m), alpha)
H_null = B_eval @ np.exp(omega_null)
#survival function of the null model
S_null = np.exp(- np.outer(H_null, np.ones(m)) )
#compute reference for ibs 
ibs_ref = np.sum((S_null - S0)**2)
# loop over the values of lambda
for l in range(len(etas)):
    eta = etas[l]
    #compute the mse for the survival function
    mse_null = np.mean((S_null-S0)**2)
    #solve rs eqs
    xi, w, v, omega = pwe.solve(eta, alpha)
    #compute ibs test
    H_test = B_eval @ np.exp(omega)
    lp_test = w * Z0 + v * Q
    S_test = np.exp( - np.outer(H_test, np.exp(lp_test)))
    ibs_test = np.sum((S_test - S0)**2)
    R_ibs = ibs_test/ibs_ref
    cv_loss = (pwe.H @ np.exp(lp_test) - C @ lp_test - pwe.F @ omega)/m
    #compute test c- index
    c_ind = c_index(T, C, lp_test)
    res = np.array([etas[l], w, v, R_ibs, c_ind, cv_loss], float)
    print(res)
    metrics[l,:] = res
df = pd.DataFrame(metrics, columns=['etas', 'w', 'v', 'R_ibs', 'c_ind', 'cv_loss'])
df.to_csv('data/rs_zeta'+"{:.2f}".format(zeta)+'_alpha'+"{:.2f}".format(alpha)+'.csv', index = False)

