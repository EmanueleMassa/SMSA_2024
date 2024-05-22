import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
zeta = 1.5
alpha = 1.0
rs_df = pd.read_csv('rs_zeta'+"{:.2f}".format(zeta)+'_alpha'+"{:.2f}".format(alpha)+'.csv')
sim_df = pd.read_csv('sim_zeta'+"{:.2f}".format(zeta)+'_alpha'+"{:.2f}".format(alpha)+'.csv')

plt.figure()
plt.errorbar(sim_df['etas'],sim_df['w_mean'],yerr =sim_df['w_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['etas'],rs_df['w'],'k-')
plt.plot(rs_df['etas'],np.ones(len(rs_df['etas'])),'k--')
plt.ylabel(r'$\hat{w}_n$')
plt.xlabel(r'$\eta$')
plt.savefig('w.jpg')

plt.figure()
plt.errorbar(sim_df['etas'],sim_df['v_mean'],yerr =sim_df['v_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['etas'],rs_df['v'],'k-')
plt.plot(rs_df['etas'],np.ones(len(rs_df['etas'])),'k--')
plt.ylabel(r'$\hat{v}_n$')
plt.xlabel(r'$\eta$')
plt.savefig('v.jpg')

plt.figure()
plt.errorbar(sim_df['etas'],sim_df['R_ibs_mean'],yerr =sim_df['R_ibs_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['etas'],rs_df['R_ibs'],'k-')
plt.plot(rs_df['etas'],np.ones(len(rs_df['etas'])),'k--')
plt.ylabel(r'$R_{IBS}$')
plt.xlabel(r'$\eta$')
plt.ylim(bottom = 0.0, top = 1.5)
plt.savefig('r_ibs.jpg')


plt.figure()
plt.errorbar(sim_df['etas'],sim_df['c_ind_mean'],yerr =sim_df['c_ind_std'],fmt = 'ko', capsize = 3)
plt.plot(rs_df['etas'],rs_df['c_ind'],'k-')
plt.ylabel(r'$HC_n$')
plt.xlabel(r'$\eta$')
plt.savefig('c_ind.jpg')

plt.show()