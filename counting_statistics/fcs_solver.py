import numpy as np
import numpy.linalg as npla
import scipy.linalg as la
import scipy.special as sp
from counting_statistics.lindblad_system import LindbladSystem
from counting_statistics import optimized_funcs

class FCSSolver(LindbladSystem):
    '''
    Users should be able to create a FCSSolver instance with the system Hamiltonian,
    Lindblad operators, somehow define the counting operators and associated rates.
    
    NEED TO MAKE COMPATIBLE WITH PYTHON 2 AND 3!

    Also need to decide how to deal with numpy Matrix objects as well as ndarrays
    
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

    def __init__(self, L, jump_op, pops, from_hilbert_space=False):
        self.L = L
        self.jump_op = jump_op
        self.pops = pops
        
        self.from_hilbert_space = from_hilbert_space
        self.__watch_variables = ['H', 'D_ops', 'D_rates', 'jump_idx', 'reduce_dim'] \
                                                if self.from_hilbert_space else ['L', 'jump_op', 'pops']
        self.__cache_is_stale = True

    @classmethod
    def from_hilbert_space(cls, H, D_ops, D_rates, jump_idx, reduce_dim=False):
        # create instance of subclass
        instance = object.__new__(cls)
        # initialize superclass first to allow construction of Liouvillian etc by LindbladSystem
        super(FCSSolver, instance).__init__(H, D_ops, D_rates, reduce_dim=reduce_dim)
        instance.jump_idx = jump_idx
        L = instance.liouvillian()
        # initialize subclass
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
        ss = v[-1].conj() / np.dot(pops, v[-1])
        return ss
    
    def mean(self):
        if self.__cache_is_stale:
            self.refresh_cache()
        return np.real(np.dot(self.pops, np.dot(self.jump_op, self.ss)))
    
    @staticmethod
    def pseudoinverse(L, freq, Q):
        return np.dot(Q, np.dot(npla.pinv(1.j*freq*np.eye(L.shape[0]) - L), Q))
    
    @staticmethod
    def Q(L, steady_state, pops):
        return np.eye(L.shape[0]) - np.outer(steady_state, pops)
    
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
        Q = self.Q(self.L, self.ss, self.pops)
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
            
        Q = self.Q(self.L, self.ss, self.pops)
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
    
    def binom_coeff_vector(self, n):
        '''Generates vector of binomial coefficients from m=1 to n, reversed.'''
        return sp.binom(n, range(n,0,-1))
    
    def generate_cumulant(self, n):
        '''Generates zero-frequency cumulant to arbitrary order using recursive scheme.
        
        Also could use a function to generate next level of hierarchy from a previously
        generated set of cumulants and states so don't need to start from the beginning
        each time.
        
        It would also be cool to dynamically generate a function for the requested cumulant
        which a user can save. Currently every time a parameter changes the cumulant needs to be regenerated which is 
        probably going to be quite inefficient for large cumulants.'''
        if self.__cache_is_stale:
            self.refresh_cache()
            
        R = self.pseudoinverse(self.L, 0, self.Q(self.L, self.ss, self.pops))
        bc_vector = self.binom_coeff_vector(n)
        cumulants = np.zeros(n)
        states = np.zeros((n+1, self.L.shape[0]), dtype='complex128')
        states[0] = self.ss
        
        def recursion(m, cumulants, states):
            # check n is an integer >= 1
            if m > 1:
                # get previous cumulants and states
                cumulants, states = recursion(m-1, cumulants, states)
            elif m == 1:
                # lowest level cumulant
                cumulants[0] =  np.dot(self.pops, np.dot(self.jump_op, states[0]))
                states[1] = np.dot(R, np.dot(cumulants[0]*np.eye(self.L.shape[0]) - self.jump_op, states[0]))
                #print states[1] + np.dot(R, np.dot(self.jump_op, states[0]))
                return cumulants, states
            else:
                raise ValueError("Cannot calculate cumulants for n < 1")
            # calculate cumulant at current level
            #cumulants[m-1] = np.dot(self.pops, np.dot(self.jump_op, np.dot(bc_vector, states[:m])))
            for i in range(m):
                cumulants[m-1] += bc_vector[i]*np.dot(self.pops, np.dot(self.jump_op, states[i]))
            # construct 3D matrix
            #W = np.vstack([bc_vector[i]*(cumulants[i]*np.eye(self.L.shape[0]) - self.jump_op)[np.newaxis,...] for i in range(m)])
            W = np.sum([np.dot(bc_vector[i]*(cumulants[i]*np.eye(self.L.shape[0]) - self.jump_op), states[i]) for i in range(m)], axis=0)
            states[m] = np.dot(R, W)
            return cumulants, states
        
        return recursion(n, cumulants, states)
    
    def generate_fano_factor(self, n):
        return self.generate_cumulant(n)[0][n-1] / self.mean()
            
            