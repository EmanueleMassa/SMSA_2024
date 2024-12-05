import numpy as np
import numpy.random as rnd
from scipy.special import lambertw
import scipy.optimize as opt
import warnings
from numba import njit 
warnings.filterwarnings('ignore')
#vectorize the lambert function
vl = np.vectorize(lambertw)

#function that computes the death and risk indicators for the sample
# @njit
def bases(t, tps):
    res1 = np.zeros((len(t),len(tps)-1))
    res2 = np.zeros((len(t),len(tps)-1))
    for k in range(len(tps)-1):
        res1[:,k] = np.array(t>=tps[k],int)*np.array(t<tps[k+1],int)
        res2[:,k] = np.minimum(t-tps[k],tps[k+1]-tps[k])* np.array(t>=tps[k],int)
    return res1, res2


#function that gives tau as a function of zeta
@njit
def inv(x, y, z, l):
    err = 1.0
    m = len(y)
    a = sum(x*y/(1.0+x*y))/m + l*x
    b = sum(y/(1.0+x*y)**2)/m + l
    while(err>1.0e-13):
        x = x - (a-z)/b
        a = sum(x*y/(1.0+x*y))/m + l*x
        b = sum(y/(1.0+x*y)**2)/m + l
        err = abs(z- a)
    return x

# @njit
def get_omega(F, B, elp, alpha):
    n = len(elp)
    a0 = F / n
    a1 = (elp @ B) / n
    res = np.zeros(len(a0))
    if(alpha == 0.0):
        res = np.log(a0 / a1)
    else:
        res = a0 / alpha - np.array( vl( np.exp(a0 / alpha) * a1 / alpha), float)
    return res
 
# @njit
def cd(c, x, F, B, alpha, eta, beta0, tol = 1.0e-8):
    err = 1.0
    its = 0
    n = len(c)
    p = len(beta0)
    elp =  np.exp(x @ beta0)
    omega0 = get_omega(F, B, elp, alpha)
    while(err >= tol):
        H0 = B @ np.exp(omega0)
        hess = H0 * elp
        s = ((hess - c) @ x ) / n
        I = (np.transpose(x) @ np.diag(hess) @ x) / n
        mu = I @ beta0
        beta1 = beta0
        for k in range(p):
            xi = I @ beta1
            phi = s[k] + xi[k] - I[k,k] * beta1[k] - mu[k]
            tau = I[k,k] + eta
            beta1[k] = -phi/tau
        elp =  np.exp(x @ beta1)
        omega1 = get_omega(F, B, elp, alpha)
        err =  np.sqrt(sum((beta1 - beta0)**2) +  sum((omega1-omega0)**2))
        beta0 = beta1
        omega0 = omega1
        its = its +1 
    return beta0, omega0

def newton(c, x, F, B, alpha, eta, beta0, tol = 1.0e-8):
    err = 1.0
    its = 0
    elp =  np.exp(x @ beta0)
    omega0 = get_omega(F, B, elp, alpha)
    p = len(beta0)
    n = len(c)
    while(err >= tol):
        H = B @ np.exp(omega0)
        ddg= H * elp
        S = ((np.transpose(x) @ np.diag(ddg) @ x) / n) + eta * np.identity(p) 
        phi = (((ddg - c) @ x) / n) + eta * beta0
        beta1 = beta0 - np.linalg.inv(S) @ phi
        lp = x @ beta1
        elp = np.exp(lp)
        omega1 = get_omega(F, B, elp, alpha)
        err = np.sqrt( sum((beta1 - beta0) ** 2) + sum((omega1 - omega0) ** 2))
        beta0 = beta1
        omega0 = omega1
        its = its+1 
    return beta0, omega0

class gauss_model:
    def __init__(self, theta, phi, rho, t1, t2, model):
        self.theta = theta
        self.phi = phi
        self.rho = rho
        self.tau1 = t1
        self.tau2 = t2
        self.model = model

    def bh(self,t):
        if(self.model == 'weibull'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))
        if(self.model == 'log-logistic'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))/(1.0+ np.exp(self.phi)*(t**self.rho))
        
    def ch(self,t):
        if(self.model == 'weibull'):
            return np.exp(self.phi)*(t**self.rho)
        if(self.model == 'log-logistic'):
            return np.log(1.0+ np.exp(self.phi)*(t**self.rho))

    def data_gen(self, n):
        #generate the data
        Z0 = rnd.normal(size = n)
        Q = rnd.normal(size = n)
        u = rnd.random(size = n)
        lp = self.theta * Z0
        if(self.model == 'weibull'):
            T1 = np.exp((np.log(-np.log(u))-lp-self.phi)/self.rho)
        if(self.model == 'log-logistic'):
            T1 = np.exp( (np.log(np.exp(-np.log(u)*np.exp(-lp))-1)-self.phi)/self.rho)
        if(self.model != 'weibull' and self.model != 'log-logistic'):
            raise TypeError("Only weibull and log-logistic are available at the moment") 
        u = rnd.random(size = n)
        T0 = (self.tau2 - self.tau1) * u + self.tau1
        T = np.minimum(T1,T0)
        C = np.array(T1<T0,int)
        #order the observations by their event times
        idx = np.argsort(T)
        T = np.array(T)[idx]
        C = np.array(C,int)[idx]
        Z0 = np.array(Z0)[idx]
        return T, C, Z0, Q

class rs_pwe:

    def __init__(self, zeta, theta0, tps, pop, H_true):
        self.theta0 = theta0
        self.tps = tps
        self.T, self.C, self.Z0, self.Q = pop
        self.b, self.B = bases(self.T, self.tps)
        self.F = self.C @ self.b
        self.m = len(self.T)
        self.H_true = H_true*np.exp(self.theta0*self.Z0)
        self.zeta = zeta
        self.w0 = 0
        self.v0 = 1.0e-3
        self.tau0 = 1.0e-3
        self.omega0 = rs_pwe.get_omega(self, np.ones(self.m), 6.0)

    def get_omega(self, xi, alpha):
        a0 = (self.F) / self.m 
        a1 = (np.exp(xi) @ self.B) / self.m
        if(alpha == 0.0):
            res = np.log(a0 / a1)
        else:
            res = a0/ alpha - np.array( vl( np.exp(a0 / alpha) * a1 / alpha),float)
        return res
    
    def solve(self, lam, alpha):
        err = 1.0
        its = 0
        eta = 0.5
        w0 = self.w0
        v0 = self.v0
        tau0 = self.tau0
        omega0 = self.omega0
        H = self.B@np.exp(omega0)
        while (err>1.0e-8):
            z = w0 * self.Z0 + v0 * self.Q + tau0 * self.C
            chi = np.array(vl(tau0 * H * np.exp(z)), float)
            xi =  z - chi 
            v1 = np.sqrt( (np.mean((tau0 * self.C - chi)**2)) / self.zeta )
            tau1 = inv(tau0, H*np.exp(xi), self.zeta, lam)
            w1 = (self.theta0 * np.mean( (tau0*self.C-chi) * (self.C-self.H_true))) / self.zeta
            omega1 = rs_pwe.get_omega(self, xi, alpha)
            v = eta * v1 + (1-eta) * v0
            w = eta * w1 + (1-eta) * w0
            tau = eta * tau1 + (1-eta) * tau0
            omega = eta * omega1 + (1.0-eta) * omega0
            err = np.sqrt((v-v0)**2 + (w-w0)**2 + (tau-tau0)**2 + (omega-omega0)@(omega-omega0))
            its = its + 1
            v0 = v
            w0 = w
            tau0 = tau
            omega0 = omega
            H = self.B @ np.exp(omega)
        self.w0 = w0
        self.v0 = v0
        self.omega0 = omega0
        self.tau0 = tau0
        self.H = H
        return xi, w0, v0, omega0




class gen_model:
    def __init__(self, A, beta, phi, rho, t1, t2, model):
        self.p = len(beta)
        self.beta = beta
        self.phi = phi
        self.rho = rho
        self.tau1 = t1
        self.tau2 = t2
        self.A = A
        self.model = model

    def bh(self, t):
        if(self.model == 'weibull'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))
        if(self.model == 'log-logistic'):
            return self.rho*np.exp(self.phi)*(t**(self.rho-1))/(1.0+ np.exp(self.phi)*(t**self.rho))
        
    def ch(self, t):
        if(self.model == 'weibull'):
            return np.exp(self.phi)*(t**self.rho)
        if(self.model == 'log-logistic'):
            return np.log(1.0+ np.exp(self.phi)*(t**self.rho))

    def gen(self, n):
        Z = rnd.normal(size = (n,self.p))
        X = Z @ self.A
        lp = X@self.beta
        u = rnd.random(size = n) 
        T0 = self.tau1 + u*(self.tau2-self.tau1)
        #sample the latent event times 
        u = rnd.random(size = n)
        if(self.model == 'weibull'):
            T1 = np.exp((np.log(-np.log(u))-lp-self.phi)/self.rho)
        if(self.model == 'log-logistic'):
            T1 = np.exp( (np.log(np.exp(-np.log(u)*np.exp(-lp))-1)-self.phi)/self.rho)
        #generate the observations 
        T = np.minimum(T1,T0)
        C = np.array(T1<T0,int)
        #order the observations by their event times
        idx = np.argsort(T)
        T = np.array(T)[idx]
        C = np.array(C,int)[idx]
        X = X[[idx],:][0,:,:]
        return T, C, X
    


class pwe_model:
    def __init__(self, p, time_points):
        self.p = p
        self.dgf = len(time_points)-1   #degrees of freedom
        self.tps = time_points          #knots
        self.d = self.dgf + self.p

    def basis(self,t):
        n = len(t)
        res = np.zeros((n,self.dgf))
        for k in range(self.dgf):
            res[:,k] = np.array(t>=self.tps[k],int)*np.array(t<self.tps[k+1],int)
        return res

    def ibasis(self,t):
        n = len(t)
        res = np.zeros((n,self.dgf))
        for k in range(self.dgf):
            ind = np.array(t>=self.tps[k],int)
            res[:,k] = np.minimum(t-self.tps[k],self.tps[k+1]-self.tps[k])*ind
        return res
    
    def fit(self, t, c, x, etas, alpha):
        self.t = t
        self.c = c
        self.x = x 
        self.n = len(t)
        self.zeta = self.p / self.n
        self.b = pwe_model.basis(self,t)
        self.B = pwe_model.ibasis(self,t)
        self.F = c @ self.b
        self.etas = etas
        betas = np.zeros((len(etas), self.p))
        omegas = np.zeros((len(etas), self.dgf))
        if(self.zeta > 1):
            self.K = self.x @ np.transpose(self.x)
            varphi = np.zeros(self.n)
        for j in range(len(etas)):
            eta = etas[j]
            if(j == 0):
                beta = np.zeros(self.p)
                omega =  get_omega(self.F, self.B, np.exp(self.x @ beta), alpha)
            # beta, omega = cd(self.c, self.x, self.F, self.B, alpha, eta, beta)
            beta, omega = newton(self.c, self.x, self.F, self.B, alpha, eta, beta)
            betas[j, :] = beta
            omegas[j, :] = omega
        self.betas = betas
        self.omegas = omegas
        return 
    
    
    
    def predict(self, t_eval, x_test):
        b_eval = pwe_model.basis(self,t_eval)
        B_eval = pwe_model.ibasis(self,t_eval) 
        ew = np.exp(np.transpose(self.omegas))
        elp = np.exp(self.betas @ np.transpose(x_test))
        h_eval = elp * np.transpose(b_eval @ ew) 
        H_eval = elp * np.transpose(B_eval @ ew) 
        S_eval = np.exp(-H_eval)
        return S_eval 
    



#function that computes the Harrel's c index
def c_index(t, c, lp):
    std_lp = np.std(lp) 
    if(std_lp <1.0e-8):
        return 0.5
    else:
        den = 0 
        c_ind = 0
        for i in range(len(t)):
            a = c * np.array(t <= t[i], int)
            den = den + sum(a)
            c_ind = c_ind + sum( a * np.array(lp > lp[i], int))
        return c_ind/den