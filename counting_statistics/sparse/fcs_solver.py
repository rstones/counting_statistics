'''
Created on 19 Jan 2017

@author: richard
'''
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class FCSSolver(object):
    
    def __init__(self, L, jump_op, pops):
        if not sp.issparse(L) or not sp.issparse(jump_op):# or not pops.issparse():
            raise ValueError("This is the sparse package so pass me some sparse matrices!")
        self.L = L
        self.jump_op = jump_op
        self.pops = pops

        self.__watch_variables = ['L', 'jump_op', 'pops']
        self.__cache_is_stale = True
        
    def refresh_cache(self):
        '''Refresh necessary quantities for counting statistics calculations.'''
        self.ss = self.stationary_state(self.L, self.pops)
        self.__cache_is_stale = False
        
    @staticmethod
    def stationary_state(L, pops):
        '''Should test for number of nullspaces found somewhere, possibly here, as the system is set up
        under the assumption it is fully connected and has a single stationary state.
        Send a warning if there are multiple nullspaces.'''
        ss = spla.eigs(L.tocsc(), k=1, sigma=None, which='SM', v0=pops/np.sum(pops))[1]
        return ss / pops.dot(ss)
    
    def mean(self):
        if self.__cache_is_stale:
            self.refresh_cache()
        return np.real(self.pops.dot(self.jump_op.dot(self.ss)))
    
    @staticmethod
    def pseudoinverse(L, Q, y):
        '''Effective calculation of pseudoinverse without need for 
        explicit calculation of pseudoinverse of L which would be dense.
        
        L is time-local generator of open system dynamics
        Q is operator projecting onto space orthogonal to steady state
        y is vector or matrix being operated on by pseudoinverse'''
        x = spla.lgmres(L, Q.dot(y))
        return x[0]
    
    @staticmethod
    def Q(L, steady_state, pops):
        return sp.eye(L.shape[0]) - sp.csr_matrix(steady_state) * sp.csr_matrix(pops.T)
    
    def zero_frequency_noise(self):
        if self.__cache_is_stale:
            self.refresh_cache()
        Q = self.Q(self.L, self.ss, self.pops)
        return np.real(self.pops.dot(self.jump_op.dot(self.ss)) \
                       - 2.*self.pops.dot(self.jump_op.dot(self.pseudoinverse(self.L, Q, self.jump_op.dot(self.ss)))))
    
    def second_order_fano_factor(self):
        return self.zero_frequency_noise() / self.mean()