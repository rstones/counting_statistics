import numpy as np
import numpy.linalg as npla
import scipy.linalg as la
from counting_statistics.lindblad_system import LindbladSystem
from counting_statistics import optimized_funcs

class FCSSolver(LindbladSystem):
    '''
    Users should be able to create a FCSSolver instance with the system Hamiltonian,
    Lindblad operators, somehow define the counting operators and associated rates.
    
    I also will need to effectively overload the __init__ function so users can provide
    a liouvillian and jump_op directly. Use @classmethod to overload constructor (sort of)
    See http://stackoverflow.com/questions/12179271/python-classmethod-and-staticmethod-for-beginner
    
    Functions: mean, zero_freq_noise, zero_freq_skewness, zero_freq_F2, zero_freq_F3,
    finite_freq_noise, finite_freq_skewness, finite_freq_F2, finite_freq_F3
    
    Class inherits from LindbladSystem which has the functionality to construct Liouvillian
    matrix from Hilbert space operators. Also has functionality to reduce dimension of the problem
    by removing equations of motion for coherences not coupled either directly or indirectly to 
    population dynamics.
    
    finite_freq functions can almost certainly be optimized with numba or cython, or at least the functions
    should be vectorized wrt the frequency values
    
    Implement caching of the steady state to save computation time. Before calculation of counting stats
    check whether state (class attributes) has changed since last function call. If not then use cached steady state
    otherwise calculate steady state again and cache it.
    
    This solver will be restricted to calculating full counting statistics for Markovian systems
    that can be expressing in Lindblad form with counting transitions occurring to a single state 
    (ie. a single drain lead in a standard electron transport setup, infinite voltage bias/unidirectional transport)
    
    Need to provide docs with references.
    
    Code should test for things like physicality of Liouvillian, is Hamiltonian Hermitian, matrix dims consistent etc...
    
    Maybe implement non-Markovian counting stats at some point.
    
    May want to include experimental support for sparse matrices, somehow minimizing need to convert to dense 
    for pinv operation. (Maybe in an FCSSolverHEOM class that extends HEOMSystem, eventually having dependence
    on the heom_solver package I will write)
    '''
    
    def __init__(self, H, D_ops, D_rates, jump_idx, reduce_dim=False):
        self.__watch_variables = ['H', 'D_ops', 'D_rates', 'jump_idx', 'reduce_dim'] # could get this with inspect
        self.__cache_is_stale = True
        
        LindbladSystem.__init__(self, H, D_ops, D_rates, reduce_dim=reduce_dim)
        self.jump_idx = jump_idx
    
    def __setattr__(self, name, value):
        '''Overloaded to listen on selected variable so cache can be refreshed.'''
        try:
            if name in self.__watch_variables:
                self.__cache_is_stale = True
        except AttributeError:
            # stop an Error being thrown when self.__watch_variables is first created on class instantiation
            # maybe throw a warning here?
            pass
        object.__setattr__(self, name, value)
        
    def refresh_cache(self):
        '''Refresh necessary quantities for counting statistics calculations.'''
        self.pops = self.I.flatten()
        self.L = self.liouvillian()
        self.ss = self.stationary_state(self.L)
        self.jump_op = self.construct_jump_operator()
        self.__cache_is_stale = False
            
    def construct_jump_operator(self):
        '''Sum kron(A,A) of all jump_ops.'''
        jump_op = np.zeros((self.sys_dim**2, self.sys_dim**2))
        for i in np.flatnonzero(self.jump_idx):
            jump_op += self.D_rates[i] * np.kron(self.D_ops[i], self.D_ops[i])
        if self.reduce_dim:
            try:
                jump_op = np.delete(jump_op, self.idx_to_remove, 0)
                jump_op = np.delete(jump_op, self.idx_to_remove, 1)
            except AttributeError:
                self.idx_to_remove = self.indices_to_remove(self.liouvillian())
        return jump_op
    
    def stationary_state(self, L):        
        # calculate
        u,s,v = la.svd(L)
        # check for number of nullspaces
        # normalize
        ss = v[-1] / np.dot(self.pops, v[-1]) # i may need to .conj() v[-1] to get correct coherences
        # cache 
        return ss
    
    def mean(self):
        if self.__cache_is_stale:
            self.refresh_cache()
        return np.real(np.dot(self.pops, np.dot(self.jump_op, self.ss)))
    
    def noise(self, freq):
        if self.__cache_is_stale:
            self.refresh_cache()
        
        # handle either array or scalar freq values
        scalar = False
        if np.isscalar(freq):
            scalar = True
            freq = np.array([freq])
        elif isinstance(freq, list):
            freq = np.array(freq)
            
        # do the calculation
        Q = np.eye(self.L.shape[0]) - np.outer(self.ss, self.pops)
        noise = np.zeros(freq.size, dtype='float64')
        for i in range(len(freq)):
            R_plus = np.dot(Q, np.dot(npla.pinv(1.j*freq[i]*np.eye(self.L.shape[0])-self.L), Q))
            R_minus = np.dot(Q, np.dot(npla.pinv(-1.j*freq[i]*np.eye(self.L.shape[0])-self.L), Q))
            noise[i] = np.dot(self.pops, np.dot(self.jump_op, self.ss)) \
                            + np.dot(self.pops, np.dot(np.dot(np.dot(self.jump_op, R_plus), self.jump_op) \
                                                       + np.dot(np.dot(self.jump_op, R_minus), self.jump_op), self.ss))
        return noise[0] if scalar else noise
    
    def skewness(self, freq_range_1, freq_range_2):
        if self.__cache_is_stale:
            self.refresh_cache()
    
    def second_order_fano_factor(self, freq):
        return self.noise(freq) / self.mean()
    
    def third_order_fano_factor(self, freq_range_1, freq_range_2):
        return self.skewness(freq_range_1, freq_range_2) / self.mean()