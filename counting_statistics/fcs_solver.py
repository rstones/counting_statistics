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
    
    def __init__(self, system_hamiltonian, lindblad_operators, lindblad_rates, reduce_dim=False):
        
        LindbladSystem.__init__(self, system_hamiltonian, lindblad_operators, lindblad_rates, reduce_dim=reduce_dim)