import numpy as np
from counting_statistics.lindblad_system import LindbladSystem

class FCSSolver(LindbladSystem):
    '''
    Users should be able to create a FCSSolver instance with the system Hamiltonian,
    Lindblad operators, somehow define the counting operators and associated rates.
    
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
    
    def __init__(self, system_hamiltonian, lindblad_operators, lindblad_rates, jump_idx, reduce_dim=False):
        self.jump_idx = jump_idx
        LindbladSystem.__init__(self, system_hamiltonian, lindblad_operators, lindblad_rates, reduce_dim=reduce_dim)
        
    def stationary_state(self):
        pass
    
    def mean(self):
        ss = self.stationary_state(self.liouvillian())
        # sum kron(A,A) of all jump_ops, there should be single row with some non-zero elements
        jump_op = np.zeros((self.sys_dim**2, self.sys_dim**2))
        for i in np.flatnonzero(self.jump_idx):
            jump_op += np.kron(self.D_ops[i], self.D_ops[i])
        # find nonzero row
        idx = np.nonzero(jump_op)
        if not np.array_equal(idx[0], idx[0]):
            raise ValueError("Jump operators represent transitions to more than one state. This is not currently supported.")
        return np.dot(self.jump_idx*self.lindblad_rates, jump_op[idx[0][0]]*ss)
    
    def noise(self, freq_range):
        ss = self.stationary_state(self.liouvillian())
    
    def skewness(self, freq_range_1, freq_range_2):
        pass
    
    def second_order_fano_factor(self, freq_range):
        return self.noise(freq_range) / self.mean()
    
    def third_order_fano_factor(self, freq_range_1, freq_range_2):
        return self.skewness(freq_range_1, freq_range_2) / self.mean()