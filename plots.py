import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
zeta_values = [0.5, 1.0, 1.5, 2.0]
alpha = 0.5
fmt = '_alpha'+"{:.2f}".format(alpha)
colors = ['r', 'g', 'b', 'y', 'c', 'm']
plt.figure()
for i in range(len(zeta_values)) :
    zeta = zeta_values[i]
    idf ='_zeta'+"{:.2f}".format(zeta) + fmt 
    rs_df = pd.read_csv('data/rs'+idf +'.csv')
    sim_df = pd.read_csv('data/sim'+idf +'.csv')
    plt.errorbar(sim_df['etas'],sim_df['w_mean'],yerr =sim_df['w_std'],fmt = colors[i] + 'o', capsize = 3)
    plt.plot(rs_df['etas'],rs_df['w'],colors[i] +'-', label = r'$\zeta = $'+"{:.2f}".format(zeta))
plt.legend()
plt.ylabel(r'$\hat{w}_n$')
plt.xlabel(r'$\eta$')
plt.savefig('figures/w' + fmt + '.jpg')

plt.figure()
for i in range(len(zeta_values)) :
    zeta = zeta_values[i]
    idf ='_zeta'+"{:.2f}".format(zeta) + fmt
    rs_df = pd.read_csv('data/rs'+idf +'.csv')
    sim_df = pd.read_csv('data/sim'+idf +'.csv')
    plt.errorbar(sim_df['etas'],sim_df['v_mean'],yerr =sim_df['v_std'],fmt = colors[i] +'o', capsize = 3)
    plt.plot(rs_df['etas'],rs_df['v'],colors[i] +'-', label = r'$\zeta = $'+"{:.2f}".format(zeta))
plt.ylabel(r'$\hat{v}_n$')
plt.xlabel(r'$\eta$')
plt.ylim(bottom = 0.0, top = 1.5)
plt.legend()
plt.savefig('figures/v' + fmt + '.jpg')

plt.figure()
for i in range(len(zeta_values)) :
    zeta = zeta_values[i]
    idf ='_zeta'+"{:.2f}".format(zeta) + fmt
    rs_df = pd.read_csv('data/rs'+idf +'.csv')
    sim_df = pd.read_csv('data/sim'+idf +'.csv')
    plt.errorbar(sim_df['etas'],sim_df['R_ibs_mean'],yerr =sim_df['R_ibs_std'],fmt = colors[i] +'o', capsize = 3)
    plt.plot(rs_df['etas'],rs_df['R_ibs'],colors[i] +'-', label = r'$\zeta = $'+"{:.2f}".format(zeta))
plt.plot(rs_df['etas'],np.ones(len(rs_df['etas'])),'k--')
plt.ylabel(r'$R_{IBS}$')
plt.xlabel(r'$\eta$')
plt.legend()
plt.savefig('figures/R_ibs' + fmt + '.jpg')

plt.figure()
for i in range(len(zeta_values)) :
    zeta = zeta_values[i]
    idf ='_zeta'+"{:.2f}".format(zeta) + fmt
    rs_df = pd.read_csv('data/rs'+idf +'.csv')
    sim_df = pd.read_csv('data/sim'+idf +'.csv')
    plt.plot(sim_df['etas'],sim_df['c_ind_mean'],colors[i] +'o')
    plt.plot(rs_df['etas'],rs_df['c_ind'],colors[i] +'-', label = r'$\zeta = $'+"{:.2f}".format(zeta))
plt.plot(rs_df['etas'],np.ones(len(rs_df['etas'])),'k-.')
plt.plot(rs_df['etas'],0.5*np.ones(len(rs_df['etas'])),'k--')
plt.ylabel(r'$HC_{{\rm test}}$')
plt.xlabel(r'$\eta$')
plt.ylim(bottom = 0.4, top = 0.8)
plt.legend()
plt.savefig('figures/c_ind' + fmt + '.jpg')

plt.show()
