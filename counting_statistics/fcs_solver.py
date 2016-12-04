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
    liouvillian, jump_op and pops directly. Use @classmethod to overload constructor (sort of)
    See http://stackoverflow.com/questions/12179271/python-classmethod-and-staticmethod-for-beginner
    
    NEED TO MAKE COMPATIBLE WITH PYTHON 2 AND 3!
    
    For the zero-frequency cumulants, I can implement a recursive scheme to generate them to arbitrary
    order following Flindt et al. 2010 (optimized using numba). Maybe worth still having up to skewness
    hard coded for speed and ease of seeing the specific structure of those equations when reading the code.
    
    finite_freq functions can almost certainly be optimized with numba or cython, or at least the functions
    should be vectorized wrt the frequency values
    
    This solver will be restricted to calculating full counting statistics for Markovian systems
    that can be expressing in Lindblad form with counting transitions occurring to a single state 
    (ie. a single drain lead in a standard electron transport setup, infinite voltage bias/unidirectional transport)
    
    Need to provide docs with references and examples in juypter notebook hosted on github.
    
    Maybe implement non-Markovian counting stats at some point.
    
    May want to include experimental support for sparse matrices, somehow minimizing need to convert to dense 
    for pinv operation. (Maybe in an FCSSolverHEOM class that extends HEOMSystem, eventually having dependence
    on the heom_solver package I will write)
    '''
    
#     def __init__(self, H, D_ops, D_rates, jump_idx, reduce_dim=False):
#         self.__watch_variables = ['H', 'D_ops', 'D_rates', 'jump_idx', 'reduce_dim'] # could get this with inspect
#         self.__cache_is_stale = True
#          
#         LindbladSystem.__init__(self, H, D_ops, D_rates, reduce_dim=reduce_dim)
#         self.jump_idx = jump_idx

    def __init__(self, time_generator, jump_op, pops, from_hilbert_space=False):
        self.from_hilbert_space = from_hilbert_space
        self.__watch_variables = ['H', 'D_ops', 'D_rates', 'jump_idx', 'reduce_dim'] \
                                                if self.from_hilbert_space else ['L', 'jump_op', 'pops']
        self.__cache_is_stale = True
        
        # instantiate liouvillian and jump here
        self.L = time_generator
        self.jump_op = jump_op
        self.pops = pops

    @classmethod
    def from_hilbert_space(cls, H, D_ops, D_rates, jump_idx, reduce_dim=False):
        # construct L and jump_op then construct instance
        instance = object.__new__(cls)
        super(FCSSolver, instance).__init__(H, D_ops, D_rates, reduce_dim=reduce_dim)
        instance.jump_idx = jump_idx
        L = instance.liouvillian()
        instance.__init__(L, instance.construct_jump_operator(L), instance.pops, from_hilbert_space=True)
        return instance
    
    def __setattr__(self, name, value):
        '''Overridden to watch selected variables to trigger cache refresh.'''
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
        if self.from_hilbert_space:
            self.pops = self.I.flatten()
            self.L = self.liouvillian()
            self.jump_op = self.construct_jump_operator(self.L)
        self.ss = self.stationary_state(self.L, self.pops)
        self.__cache_is_stale = False

    def construct_jump_operator(self, L):
        '''Sum kron(A,A) of all jump_ops.'''
        jump_op = np.zeros((self.sys_dim**2, self.sys_dim**2))
        for i in np.flatnonzero(self.jump_idx):
            jump_op += self.D_rates[i] * np.kron(self.D_ops[i], self.D_ops[i])
        if self.reduce_dim:
            try:
                jump_op = np.delete(jump_op, self.idx_to_remove, 0)
                jump_op = np.delete(jump_op, self.idx_to_remove, 1)
            except AttributeError:
                self.idx_to_remove = self.indices_to_remove(L)
                jump_op = np.delete(jump_op, self.idx_to_remove, 0)
                jump_op = np.delete(jump_op, self.idx_to_remove, 1)
        return jump_op
    
    @staticmethod
    def stationary_state(L, pops):
        '''Should test for number of nullspaces found somewhere, possibly here, as the system is set up
        under the assumption it is fully connected and has a single stationary state.
        Send a warning if there are multiple nullspaces.''' 
        # calculate
        u,s,v = la.svd(L)
        # check for number of nullspaces
        # normalize
        ss = v[-1].conj() / np.dot(pops, v[-1]) # i may need to .conj() v[-1] to get correct coherences
        return ss
    
    def mean(self):
        if self.__cache_is_stale:
            self.refresh_cache()
        return np.real(np.dot(self.pops, np.dot(self.jump_op, self.ss)))
    
    @staticmethod
    def pseudoinverse(L, freq, Q):
        return np.dot(Q, np.dot(npla.pinv(1.j*freq*np.eye(L.shape[0]) - L), Q))
    
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
        noise = np.zeros(freq.size, dtype='complex128')
        for i in range(len(freq)):
            R_plus = self.pseudoinverse(self.L, freq[i], Q)
            R_minus = self.pseudoinverse(self.L, -freq[i], Q)
            noise[i] = np.dot(self.pops, np.dot(self.jump_op \
                            + np.dot(np.dot(self.jump_op, R_plus), self.jump_op) \
                                        + np.dot(np.dot(self.jump_op, R_minus), self.jump_op), self.ss))
        return np.real(noise[0] if scalar else noise)
    
    def skewness(self, freq1, freq2):
        if self.__cache_is_stale:
            self.refresh_cache()
            
        Q = np.eye(self.L.shape[0]) - np.outer(self.ss, self.pops)
        skewness = np.zeros((freq1.size, freq2.size), dtype='complex128')
        for i in range(len(freq1)):
            for j in range(len(freq2)):
                '''Currently ignoring zero-frequency limit as its a bit more complicated
                than for the noise. This should cause a test failure until its fixed.'''
                if freq1[i] == 0 or freq2[j] == 0 or freq1[i] == freq2[j]:
                    continue
                R1 = self.pseudoinverse(self.L, -freq1[i], Q)
                R2 = self.pseudoinverse(self.L, freq1[i]-freq2[j], Q)
                R3 = self.pseudoinverse(self.L, freq2[j], Q)
                R4 = self.pseudoinverse(self.L, -freq2[j], Q)
                R5 = self.pseudoinverse(self.L, freq1[i], Q)
                R6 = self.pseudoinverse(self.L, freq2[j]-freq1[i], Q)
                jump_op_average = np.dot(self.pops, np.dot(self.jump_op, self.ss))
                skewness[i,j] = np.dot(self.pops, np.dot(self.jump_op \
                        + np.dot(self.jump_op, np.dot(R1+R2+R3, self.jump_op)) \
                        + np.dot(self.jump_op, np.dot(R4+R5+R6, self.jump_op)) \
                        + np.dot(np.dot(self.jump_op, R1), np.dot(self.jump_op, np.dot(R4+R6, self.jump_op))) \
                        + np.dot(np.dot(self.jump_op, R2), np.dot(self.jump_op, np.dot(R4+R5, self.jump_op))) \
                        + np.dot(np.dot(self.jump_op, R3), np.dot(self.jump_op, np.dot(R5+R6, self.jump_op))) \
                        + (-jump_op_average/(1.j*freq1[i])) * np.dot(self.jump_op, np.dot(R4-R2+R6-R3, self.jump_op)) \
                        + (jump_op_average/(1.j*freq1[i]-1.j*freq2[j])) * np.dot(self.jump_op, np.dot(R4-R1+R5-R3, self.jump_op)) \
                        + (jump_op_average/(1.j*freq2[j])) * np.dot(self.jump_op, np.dot(R6-R1+R5-R2, self.jump_op)), self.ss))
        return np.real(skewness)
                
    def second_order_fano_factor(self, freq):
        return self.noise(freq) / self.mean()
    
    def third_order_fano_factor(self, freq1, freq2):
        return self.skewness(freq1, freq2) / self.mean()