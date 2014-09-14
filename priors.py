"""Source code for priors project

Key notation:
    T: Number of observations
    K: Number of covariates 
    M: Parameter support points
    J: Error support points
    y: Dependent variable (T-by-1)
    X: Covariate matix (T-by-K)
    q: Prior on signal (KM-by-1)
    u: Prior on noise (TJ-by-1)
    z: Parameter support vector (M-by-1)
    v: Noise support vector (J-by-1)
    lmda: Lagrange multipliers (T-by-1)
            
"""

import numpy as np
from itertools import permutations, product
from scipy.misc import logsumexp
from scipy.optimize import minimize
import pandas as pd
#import matplotlib # use if on HPC
#matplotlib.use('Agg') # use if on HPC
import matplotlib.pyplot as plt
import time

class Prior(object):
    def __init__(self, T, N, parms, a, b, corr, rho, proc, sd, z, x0, path):
        """Initializes Prior class."""
        K = len(parms[1]) # number of covariates
        M = J = len(z) # number of parameter and error support points
        X = self.X = self.xdata(T, K, a, b, corr)
        z = np.array(z)[:, np.newaxis]
        u = (1.0/J)*np.ones(T*J)[:, np.newaxis] # uniform prior on noise
        priors = self.priors(M, rho, proc)     
        # run experiment
        self.experiment(N, M, X, u, z, priors, parms, sd, x0, corr, proc, path)    
        # calculate results
        self.results(path, T, sd, corr, parms, z, proc)
        # create graphs
        self.graphs(path, T, sd, corr, parms, z, proc)
                
    def experiment(self, N, M, X, u, z, priors, parms, sd, x0, corr, proc, path):
        """Returns experiment results.
        
        Parameters
        ----------
        N : int
            Number of replications
        M : int
            Number of error support points
        X : ndarray
            Covariate matrix
        u : ndarray
            Prior on noise
        z : ndarray
            Parameter support vector
        priors : tuple
            Possible priors
        parms : tuple
            Model parameters
        sd : int
            Standard deviation on noise to generate dependent variable
        x0 : ndarray
            Starting values
        corr : int
            Correlated covariates
        proc : int
            Number of coefficients receiving prior procedure 
        path : str
            File path
        
        """
        assert type(N) == type(M) == int, 'N and M must be integers'
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        assert type(sd) == int, 'sd must be integer'
        assert corr==0 or corr==1, 'corr misspecified' 
        assert proc==1 or proc==2, 'proc is misspecified'
        T, K = np.shape(X)
        for i in range(N):
            print 'Enter replication {}'.format(i + 1)
            t0 = time.time()
            y = self.ydata(X, parms[1], sd)
            v = self.esupport(y, M)
            # define objective function as maximization of dual
            obj = lambda lmda, q: - self.dual(lmda, q, u, y, X, z, v) 
            # define Jacobian
            jac = lambda lmda, q: - self.jacobian(lmda, q, u, y, X, z, v)
            coeff, ols, dev_ent, dev_ols, ent, success = self.fit_model(obj, 
                jac, priors[0], x0, y, X, u, z, v, parms[1], proc)   
            data = np.hstack((np.vstack((priors[0])), np.vstack((priors[1])),
                np.vstack((coeff)), np.vstack((ols)), np.vstack((dev_ent)), 
                np.vstack((dev_ols)), np.vstack((ent)), np.vstack((success))))  
            self.out(path, T, M, parms, sd, corr, data, i, z, proc)  
            print 'Exit replication {0} ({1} seconds wall time)'.format(i + 1,
                time.time() - t0 )
        
    def fit_model(self, obj, jac, priors, x0, y, X, u, z, v, parms, proc):
        """Minimizes dual objective function. 
    
        Parameters
        ----------
        obj : callable func
            Objective function
        jac : callable func
            Jacobian of objective function
        priors : ndarray
            Possible priors
        x0 : ndarray
            Starting values
        y : ndarray
            Dependent variable
        X : ndarray
            Covariates
        u : ndarray
            Prior on noise
        z : ndarray
            Parameter support vector
        v : ndarray  
            Noise support vector  
        parms : list
            Model parameters  
        proc : int
            Number of coefficients receiving prior procedure     
    
        """ 
        assert np.shape(y)[1] == 1, 'y dimension issue'  
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        assert np.shape(v)[1] == 1, 'v dimension issue'
        assert proc==1 or proc==2, 'proc is misspecified'
        T = len(y)
        K = np.shape(X)[1]
        M = len(z)
        q = (1.0/M)*np.ones(K*M)[:, np.newaxis] 
        maxiter=900 if sd>1 else 90 + 360*corr # slack to large sd/corr
        coeffs, olss, dev_ents, dev_olss, ents, success = [],[],[],[],[],[]
        for prior in priors:
            toprint = [round(e, 2) for e in prior]
            print 'Estimating model with prior: {}'.format(toprint)
            # introduce iteration-specific prior into q
            q[M: (proc + 1)*M] = np.array(prior)[:, np.newaxis]
            assert np.shape(q)[1] == 1, 'q dimension issue' 
            # fit model
            result = minimize(obj, x0, args=(q,), method='CG', jac=jac,
                tol=1e-06, options={'maxiter': maxiter}) 
            assert np.isfinite(result.x).all() == True, 'result.x not finite'
            if result.success==True: # check derivatives only if successful
                jchk = self.jacobian(result.x, q, u, y, X, z, v)
                assert np.allclose(jchk, 0, atol=1e-05) == True, 'jacobian issue'
            print 'Optimizer exited successfully: {}'.format(result.success)
            # get output
            pprob = self.pprobs(result.x, X, q, z)
            wprob = self.wprobs(result.x, u, v, T)       
            coeff = self.coeffs(pprob, z)
            ols = self.ols(y, X)
            dev_ent = self.sq_dev(parms, coeff.T)
            dev_ols = self.sq_dev(parms, ols.T)
            ent = self.ce(pprob, wprob, q, u)           
            coeffs.append(coeff)
            olss.append(ols)           
            dev_ents.append(dev_ent)
            dev_olss.append(dev_ols)
            ents.append(ent)
            success.append(int(result.success))
        return coeffs, olss, dev_ents, dev_olss, ents, success
        
    def ols(self, y, X):
        """Returns OLS estimates.
        
        Parameters
        ----------
        y : ndarray
            Dependent variable
        X : ndarray
            Covariates
            
        """
        assert np.shape(y)[1] == 1, 'y dimension issue' 
        p1 = np.linalg.inv(np.dot(X.T, X))
        p2 = np.dot(X.T, y)
        ols = np.dot(p1, p2)       
        assert np.isfinite(ols).all() == True, 'ols not finite'
        return ols.T

    def dual(self, lmda, q, u, y, X, z, v):
        """Returns value of dual objective function.
    
        Parameters
        ----------
        lmda : ndarray
            Lagrange multipliers
        q : ndarray
            Prior on signal
        u : ndarray
            Prior on noise
        y : ndarray
            Dependent variable  
        X : ndarray
            Covariate matrix
        z : ndarray
            Parameter support vector
        v : ndarray  
            Noise support vector  
    
        """
        assert np.shape(q)[1] == 1, 'q dimension issue'
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(y)[1] == 1, 'y dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        assert np.shape(v)[1] == 1, 'v dimension issue'
        lmda = lmda[:, np.newaxis]
        T = np.shape(X)[0]
        K = np.shape(X)[1]
        M = np.shape(z)[0]
        J = np.shape(v)[0]
        eye_k = np.eye(K)
        eye_t = np.eye(T)
        Z = np.kron(eye_k, z.T)
        V = np.kron(eye_t, v.T)
        p1 = np.dot(y.T, lmda)[0][0] 
        assert type(p1) == np.float64, 'p1 must be float'
        p2a = np.dot(Z.T, np.dot(X.T, lmda)) # KM-by-1
        p2b = np.reshape(p2a, (-1, M))
        p2c = logsumexp(p2b, axis=1, b=np.reshape(q, (-1, M)))
        p2 = np.sum(p2c)  
        assert type(p2) == np.float64, 'p2 must be float'
        assert np.isfinite(p2).all() == True, 'p2 not finite'
        p3a = np.dot(V.T, lmda) # TJ-by-1   
        p3b = np.reshape(p3a, (-1, J))
        p3c = logsumexp(p3b, axis=1, b=np.reshape(u, (-1, J)))
        p3 = np.sum(p3c)
        assert type(p3) == np.float64, 'p3 must be float'
        assert np.isfinite(p3).all() == True, 'p3 not finite'
        return p1 - p2 - p3
        
    def jacobian(self, lmda, q, u, y, X, z, v):   
        """Returns Jacobian of dual formulation.

        Parameters
        ----------
        lmda: array
            Lagrange multipliers           
        q : ndarray
            Prior on signal
        u : ndarray
            Prior on noise
        y : ndarray
            Dependent variable  
        X : ndarray
            Covariate matrix
        z : ndarray
            Parameter support vector
        v : ndarray  
            Noise support vector  

        """
        assert np.shape(q)[1] == 1, 'q dimension issue'
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(y)[1] == 1, 'y dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        assert np.shape(v)[1] == 1, 'v dimension issue'
        T = np.shape(X)[0]
        K = np.shape(X)[1]      
        eye_t = np.eye(T)
        eye_k = np.eye(K)
        Z = np.kron(eye_k, z.T)
        V = np.kron(eye_t, v.T)
        pprob = self.pprobs(lmda, X, q, z)
        wprob = self.wprobs(lmda, u, v, T)
        return np.squeeze(y - np.dot(np.dot(X, Z), pprob) - np.dot(V, wprob))
              
    def ce(self, pprob, wprob, q, u):
        """Returns cross entropy.
    
        Parameters
        ----------
        pprob : ndarray
            Probabilities on signal
        wprob : ndarray
            Probabilities on noise
        q : ndarray
            Prior on signal 
        u : ndarray
            Prior on noise
    
        """
        assert np.shape(pprob)[1] == 1, 'pprob dimension issue' 
        assert np.shape(wprob)[1] == 1, 'wprob dimension issue'      
        assert np.shape(q)[1] == 1, 'q dimension issue' 
        assert np.shape(u)[1] == 1, 'u dimension issue'          
        assert len(pprob) == len(q), "len(pprob) not equal to len(q)"  
        assert len(wprob) == len(u), "len(wprob) not equal to len(u)"          
        ce_parm  = np.sum(pprob[3:6] * np.log((pprob[3:6] + 1e-10)/q[3:6]))              
        ce_signal = np.sum(pprob * np.log((pprob + 1e-10)/q))
        ce_noise = np.sum(wprob * np.log((wprob + 1e-10)/u))
        ce_total = ce_signal + ce_noise
        assert type(ce_parm) == np.float64, 'ce_parm must be float'
        assert type(ce_signal) == np.float64, 'ce_signal must be float'
        assert type(ce_noise) == np.float64, 'ce_noise must be float'
        assert type(ce_total) == np.float64, 'ce_total must be float'
        return np.array([[ce_parm, ce_signal, ce_noise, ce_total]])

    def pprobs(self, lmda, X, q, z):
        """Returns probabilities on signal.
    
        Parameters
        ----------
        lmda : ndarray
            Lagrange multipliers
        X : ndarray
            Covariate matrix
        q : ndarray
            Prior on signal
        z : ndarray
            Parameter support vector 
    
        """
        assert np.shape(q)[1] == 1, 'q dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        lmda = lmda[:, np.newaxis]
        K = np.shape(X)[1]
        M = np.shape(z)[0]
        eye_k = np.eye(K)
        Z = np.kron(eye_k, z.T)
        logprobs1 = np.log(q) + np.dot(Z.T, np.dot(X.T, lmda))
        logprobs2 = logsumexp(logprobs1.reshape(K, M), axis=1)
        logprobs3 = np.repeat(logprobs2, M)[:, np.newaxis]
        probs = np.exp(logprobs1 - logprobs3)
        assert np.isfinite(probs).all() == True, 'probs not finite'
        assert np.shape(probs) == (K*M, 1), 'probs dimension issue'
        return probs        
    
    def wprobs(self, lmda, u, v, T):
        """Returns probabilities on noise.
    
        Parameters
        ----------
        lmda : ndarray
            Lagrange multipliers
        u : ndarray
            Prior on noise
        v : ndarray  
            Noise support vector 
        T : int
            Number of observations
            
        """  
        assert np.shape(u)[1] == 1, 'u dimension issue'
        assert np.shape(v)[1] == 1, 'v dimension issue'
        assert type(T) == int, 'T must be integer'
        lmda = lmda[:, np.newaxis]
        J = np.shape(v)[0]
        eye_t = np.eye(T)
        V = np.kron(eye_t, v.T)     
        logprobs1 = np.log(u) + np.dot(V.T, lmda)        
        logprobs2 = logsumexp(logprobs1.reshape(T, J), axis=1)       
        logprobs3 = np.repeat(logprobs2, J)[:, np.newaxis]
        probs = np.exp(logprobs1 - logprobs3)
        assert np.isfinite(probs).all() == True, 'probs not finite'
        assert np.shape(probs) == (T*J, 1), 'probs dimension issue'
        return probs
        
    def xdata(self, T, K, a, b, corr):
        """Returns covariate data.
    
        Parameters
        ----------
        T : int
            Number of observations
        K : int
            Number of covariates
        a : float
            Mean in normal distribution of covariates
        b : float
            Standard deviation in normal distribution of covariates
        corr : int
            Correlated covariates
    
        """
        assert T > 0 and K > 0, 'inputs must be non-negative'
        assert b > 0, 'b must be non-negative'
        assert corr==0 or corr==1, 'corr misspecified'      
        mu=1.0 if corr==0 else 100.0 # specify condition number  
        temp = np.random.normal(a, b, size=(T, K))
        U, s, V = np.linalg.svd(temp, full_matrices=False)
        snew = np.ones(K)
        snew[K - 1] = np.sqrt(2.0/(1 + mu))
        snew[0] = np.sqrt(2.0*mu/(1 + mu))
        X = np.dot(U, np.dot(np.diag(snew), V))
        cn = np.linalg.cond(np.dot(X.T, X))
        assert np.allclose(mu, cn), 'condition number issue' 
        assert np.shape(X) == (T, K), 'X dimension issue'
        return X
    
    def ydata(self, X, parms, sd):
        """Returns dependent variable.
    
        Parameters
        ----------
        X : ndarray
            Covariates
        parms : list
            Model parameters
        sd : int
            Standard deviation on noise
    
        """
        assert sd > 0, 'sd must be greater than zero'
        parms = np.atleast_2d(parms)
        T = np.shape(X)[0] # number of observations
        e = np.random.normal(0, sd, size=(T, 1)) # noise
        assert np.shape(X)[1] == np.shape(parms)[1], 'array shape mismatch'
        y = np.dot(X, parms.T) + e
        assert np.shape(y) == (T, 1), 'y dimension issue'
        return y
        
    def priors(self, M, rho, proc):
        """Returns prior choices.
    
        Parameters
        ----------
        M : int
            Number of support points
        rho : float
            Dispersion control parameter
        proc : int
            Number of coefficients receiving prior procedure
        
        """
        assert M > 0, 'M must be greater than zero'
        assert type(M) == int, 'M must be an integer'
        assert proc==1 or proc==2, 'proc is misspecified'
        seq = xrange(1, M + 1)
        prior = []
        for perm in permutations(seq, M):
            temp = np.asarray(perm)**rho
            prior.append(temp / np.sum(temp))                           
        prior.append(np.ones(M) / M)   # add uniform prior    
        labels = range(1, len(prior) + 1)       
        if proc==2: # conduct prior procedure on two coefficients
            prod1 = product(prior, prior)
            prod2 = product(labels, labels)
            prior = [np.concatenate(i) for i in prod1]
            labels = [10*i[0] + i[1] for i in prod2]
        return prior, labels

    def esupport(self, y, M):
        """Returns error support vector.
    
        Parameters
        ----------
        y : ndarray
            Dependent variable
        M : int
            Number of support points
        
        """
        assert np.shape(y)[1] == 1, 'y dimension issue'
        assert type(M) == int, 'M must be an integer'
        assert M==3 or M==5, "M must equal 3 or 5"
        sig = np.std(y)
        if M==3:
            v = np.array([-3*sig, 0, 3*sig])[:, np.newaxis]
        else: # M=5
            v = np.array([-3*sig, -1.5*sig, 0, 1.5*sig, 3*sig])[:, np.newaxis]
        return v
        
    def coeffs(self, probs, z):
        """Returns coefficient estimates.
        
        Parameters
        ----------
        probs : ndarray
            Probabilities on signal
        z : ndarray
            Parameter support vector
            
        """
        assert np.shape(probs)[1] == 1, 'probs dimension issue'
        assert np.shape(z)[1] == 1, 'z dimension issue'
        K = len(probs)/len(z)
        eye_k = np.eye(K)
        Z = np.kron(eye_k, z.T)   
        assert np.shape(Z) == (K, K*len(z)), 'Z dimension issue'
        return np.dot(Z, probs).T
    
    def sq_dev(self, parms, coeffs):
        """Returns sum of squared deviations of estimates from true values.
        
        Parameters
        ----------
        parms : list
            Model parameters
        coeffs : ndarray
            Estimated coefficients
            
        """
        assert type(coeffs) == np.ndarray, 'coeffs must be array'        
        assert np.shape(coeffs)[1] == 1, 'coeffs dimension issue'
        assert len(parms) == len(coeffs), 'len(parms) not equal to len(coeffs)'
        parms = np.asarray(parms)[:, np.newaxis]        
        sq_dev = (parms - coeffs)**2
        sq_dev = np.vstack((sq_dev, sq_dev.sum())) # add total squared dev.
        assert np.shape(sq_dev)[1] == 1, 'sq_dev dimension issue'
        return sq_dev.T 
 
    def out(self, path, T, M, parms, sd, corr, data, n, z, proc):
        """Writes model output to file. 
        
        Parameters
        ----------
        path : str
            File path
        T : int
            Number of observations
        M : int
            Number of error support points
        parms : tuple
            Model parameters
        sd : int
            Standard deviation on noise to generate dependent variable
        corr : int
            Correlated covariates
        data : ndarray
            Data to be written to file
        n : int
            Replication number
        z : ndarray
            Parameter support vector
        proc : int
            Number of coefficients receiving prior procedure 
            
        """
        assert type(T)==type(M)==type(sd)==int, 'T, M, and sd must be integers'
        assert corr==0 or corr==1, 'corr misspecified' 
        assert proc==1 or proc==2, 'proc is misspecified'
        nstr1 = str(parms[0]) + str(T) + str(sd)
        nstr2 = str(corr) + str(int(z[-1][0])) + str(proc)
        nstr = nstr1 + nstr2 
        fname = path + nstr + 'out' + '.csv'
        prior, b, ols, dev_ent, dev_ols = [], [], [], [], []  
        if n==0: # write header and data
            with open(fname, 'w') as f:
                # write header
                for i in range(proc*M):
                    prior.append('prior' + str(i))
                for i in range(len(parms[1])):
                    b.append('b' + str(i))
                    ols.append('ols' + str(i))
                    dev_ent.append('dev_ent' + str(i))
                    dev_ols.append('dev_ols' + str(i))
                dev_ent.append('dev_ent')
                dev_ols.append('dev_ols')
                ce = ['ce_parm', 'ce_signal', 'ce_noise', 'ce_total']    
                header = prior + ['labels'] + b + ols + dev_ent + dev_ols + ce \
                    + ['success']
                f.write(','.join(header))
                # write data
                for i in data:
                    strdata = [str(j) for j in i]
                    f.write('\n' + ','.join(strdata))
        else: # just write data
            with open(fname, 'a') as f:
                for i in data:
                    strdata = [str(j) for j in i]
                    f.write('\n' + ','.join(strdata))
                    
    def results(self, path, T, sd, corr, parms, z, proc):
        """Calculates post-simulation results.
        
        Parameters
        ----------
        path : str
            File path
        T : int
            Number of observations
        sd : int
            Standard deviation on noise to generate dependent variable
        corr : int
            Correlated covariates
        parms : tuple
            Model parameters
        z : ndarray
            Parameter support vector
        proc : int
            Number of coefficients receiving prior procedure 
        
        """
        assert type(T) == type(sd) == int, 'T and sd must be integers' 
        assert proc==1 or proc==2, 'proc is misspecified'         
        nstr1 = str(parms[0]) + str(T) + str(sd)
        nstr2 = str(corr) + str(int(z[-1][0])) + str(proc)
        nstr = nstr1 + nstr2            
        data = path + nstr + 'out'  + '.csv'
        mresults = path + nstr + 'mresults' + '.csv'
        vresults = path + nstr + 'vresults' + '.csv'
        presults = path + nstr + 'presults' + '.csv' 
        df = pd.read_csv(data) 
        # delete unsuccessful entries while preserving OLS estimates
        df.ix[:,'b0':'b2'][df['success']==0] = np.nan
        df.ix[:,'dev_ent0':'dev_ent'][df['success']==0] = np.nan
        df.ix[:,'ce_signal':'success'][df['success']==0] = np.nan        
        # group by prior
        prior = [i for i in df.columns if i.startswith('prior')]
        grouped = df.groupby(prior) 
        means = grouped.agg(np.mean) # aggregate by means
        variances = grouped.agg(np.var) # aggregate by variances
        cnames = ['dev_ent','dev_ent1','ce_signal','ce_noise','ce_total']
        correlations = means.ix[:, cnames].corr(method='spearman')
        # bias calculations 
        for i in range(len(parms[1])): # for ce coefficients
            bias_ent = 'bias_ent' + str(i)
            b = 'b' + str(i)
            means[bias_ent] = means[b] - parms[1][i]
        for i in range(len(parms[1])): # for ols coefficients
            bias_ols = 'bias_ols' + str(i)
            ols = 'ols' + str(i)
            means[bias_ols] = means[ols] - parms[1][i]
        means.to_csv(mresults) # write to csv
        variances.to_csv(vresults) 
        correlations.to_csv(presults)
                      
    def graphs(self, path, T, sd, corr, parms, z, proc): 
        """Calculates post-simulation graphs.
        
        Parameters
        ----------
        path : str
            File path
        T : int
            Number of observations
        sd : int
            Standard deviation on noise to generate dependent variable
        corr : int
            Correlated covariates
        parms : tuple
            Model parameters
        z : ndarray
            Parameter support vector
        proc : int
            Number of coefficients receiving prior procedure 
        
        """  
        assert type(T) == type(sd) == int, 'T and sd must be integers' 
        assert proc==1 or proc==2, 'proc is misspecified'            
        nstr1 = str(parms[0]) + str(T) + str(sd)
        nstr2 = str(corr) + str(int(z[-1][0])) + str(proc)
        nstr = nstr1 + nstr2 
        data = path + nstr + 'out'  + '.csv'
        figure = path + nstr + 'figure'    
        df_raw = pd.read_csv(data) 
        df = df_raw[df_raw['success']==1] # delete unsuccessful entries
        # group by prior
        prior = [i for i in df.columns if i.startswith('prior')]
        grouped = df.groupby(prior) 
        means = grouped.agg(np.mean) # aggregate by means 
        lst = [('dev_ent1','ce_parm', 'all'),('dev_ent1','ce_parm','means'),
            ('dev_ent','ce_parm','all'),('dev_ent','ce_parm','means'),
            ('dev_ent','ce_signal', 'all'),('dev_ent','ce_signal','means'),
            ('dev_ent','ce_total','all'),('dev_ent','ce_total','means')]                 
        npriors = len(means)
        marker = [tuple(np.random.uniform(0, 1, size=(1, 3))[0]) 
            for i in range(npriors)] # list of RGB tuples   
        for i in lst:
            plt.figure()
            plt.xlabel('Mean Squared Error')
            plt.ylabel('Cross Entropy')
            if i[2]=='means':
                plt.scatter(means[i[0]], means[i[1]], marker='o', c='k')
                if len(prior)==3:
                    for j in range(npriors):
                        plt.annotate(int(means['labels'].iloc[j]), 
                            (means[i[0]].iloc[j], means[i[1]].iloc[j]), 
                            size='small', xytext=(0, 5), 
                            textcoords='offset points')
                plt.savefig(figure + '(' + i[0] + ')' + '(' + i[1] + ')' +
                    '(' + i[2] + ')' + '.png', bbox_inches='tight')
                if len(prior) > 3:   
                    plt.figure()
                    best = means.sort('dev_ent')[0: 20] 
                    plt.scatter(best[i[0]], best[i[1]], marker='o', c='k')
                    for j in range(20):
                        plt.annotate(int(best['labels'].iloc[j]), 
                            (best[i[0]].iloc[j],best[i[1]].iloc[j]), 
                            size='small', xytext=(-15,0), 
                            textcoords='offset points')
                    plt.savefig(figure + '(' + i[0] + ')' + '(' + i[1] + ')' + 
                        '(' + i[2] + ')' + '(' + '2' + ')'+ '.png', 
                        bbox_inches='tight')
            else:
                counter=0    
                for key, grp in grouped:                       
                    temp1 = [round(j, 2) for j in key] # two decimal places               
                    temp2 = str(tuple(temp1))
                    plt.scatter(grp[i[0]], grp[i[1]], marker='o',
                        c=marker[counter], label=temp2) 
                    counter += 1  
                if len(prior)==3: 
                    lgd = plt.legend(scatterpoints=1, 
                        bbox_to_anchor=(1.25,1), fontsize='x-small', ncol=1)   
                else:
                    lgd = plt.legend(scatterpoints=1, 
                        bbox_to_anchor=(2.22,1), fontsize='x-small', ncol=3)   
                plt.savefig(figure + '(' + i[0] + ')' + '(' + i[1] + ')' + 
                    '(' + i[2] + ')' + '.png', bbox_extra_artists=(lgd,), 
                    bbox_inches='tight')                                 
                                 
if __name__ == "__main__":

    # set seed
    np.random.seed(123)

    # user inputs
    T = 50 # sample size: [10, 20, 50, 100, 200]
    N = 1000 # number of replications
    parms_menu = [(0, [1., -3., 2.]),
                  (1, [1., -6., 2.]),
                  (2, [2., -6., 4.]),
                  (3, [1., -3., 2., 0., 4., -1., -2.]),
                  (4, [1., -6., 2., 0., 4., -1., -2.]), 
                  (5, [2., -6., 4., 0., 8., -2., -4.])]
    parms = parms_menu[2] # parameters values
    a = 0 # mean in normal distribution of covariates
    b = 1 # standard deviation in normal distribution of covariates
    corr = 1 # correlated covariates: [0, 1]
    rho = 0.5 # dispersion control parameter
    proc = 1 # number of coefficients receiving prior procedure: [1, 2]
    sd = 1 # standard deviation on model noise: [1, 5]
    z = [-10., 0., 10.] # support for parameters
    x0 = np.zeros(T) # starting values
    path = '/Users/hendersonhl/Documents/Articles/Optimal-Prior/Output/'
    #path = '/home/hh9467a/Output/'
    
    # run experiment
    exp = Prior(T, N, parms, a, b, corr, rho, proc, sd, z, x0, path)

