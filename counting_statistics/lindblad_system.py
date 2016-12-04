import numpy as np
from counting_statistics.physicality_error import PhysicalityError

class LindbladSystem(object):
    '''Class to construct a Liouville space operator representation of a Lindblad master equation.'''
    
    def __init__(self, system_hamiltonian, lindblad_operators, lindblad_rates, reduce_dim=False):
        
        if not self.is_hermitian(system_hamiltonian):
            raise PhysicalityError("System Hamiltonian is not Hermitian!")
        self.H = system_hamiltonian
        self.sys_dim = self.H.shape[0]
        self.I = np.eye(self.sys_dim)
        self.pops = self.I.flatten() # mask array indicating the position of population elements with element 1
        
        if len(lindblad_operators) != len(lindblad_rates):
            raise ValueError("Different number of Lindblad operators and rates provided.")
        self.D_ops = lindblad_operators
        self.D_rates = lindblad_rates
        
        self.reduce_dim = reduce_dim
    
    @classmethod
    def from_hilbert_space(cls, system_hamiltonian, lindblad_operators, lindblad_rates, reduce_dim=False):
        # construct liouvillian and jump operator
        
        # return instance of the base and inheriting class
        return cls()
    
    def liouvillian(self):
        '''Constructs full Liouvillian of the system.'''
        L = -1.j * (np.kron(self.H, self.I) - np.kron(self.I, self.H)) + self.lindblad_dissipator()
        self.pops = self.I.flatten() # refresh self.pops in case it has been previously reduced
        return self.reduced_liouvillian(L) if self.reduce_dim else L
    
    def lindblad_dissipator(self):
        '''Constructs Lindblad dissipator in Liouville space.'''
        D = np.zeros((self.sys_dim**2, self.sys_dim**2), dtype='complex128')
        for i in range(len(self.D_ops)):
            A = self.D_ops[i]
            A_dagger_A = np.dot(A.conj().T, A)
            D += self.D_rates[i] * (np.kron(A, A) - 0.5 * (np.kron(A_dagger_A, self.I) + np.kron(self.I, A_dagger_A)))
        if not self.is_dissipator_physical(D, self.pops):
            raise PhysicalityError("Lindblad dissipator is non-physical! Dynamics may be non-trace preserving.")
        return D
    
    def reduced_liouvillian(self, L):
        '''Reduces dimension of Liouvillian by removing rows and columns corresponding
        to the dynamics of coherences that are decoupled from populations. Also removes
        elements from self.pops to keep dimensions consistent.
        '''
        self.idx_to_remove = self.indices_to_remove(L)
        L = np.delete(L, self.idx_to_remove, 0)
        L = np.delete(L, self.idx_to_remove, 1)
        self.pops = np.delete(self.pops, self.idx_to_remove, 0)
        return L
    
    def indices_to_remove(self, L):
        '''Finds the row indices of decoupled coherences in the Liouvillian.'''
        # get indices of population elements
        indices_to_keep = [np.flatnonzero(self.pops).tolist()]
        # get indices of coherences directly coupled to populations
        indices_to_keep.append([])
        for i,row in enumerate(L):
            if i not in indices_to_keep[-2] and (row*self.pops).any():
                indices_to_keep[-1].append(i)
        
        # recursively find coherences coupled to populations indirectly via other coherences
        counter = -3
        while True:
            indices_to_keep.append([])
            kept_rows = [i for l in indices_to_keep for i in l]
            for i,row in enumerate(L):
                if i not in kept_rows:
                    for j,el in enumerate(indices_to_keep[-2]):
                        if (row*L[el]).any():
                            indices_to_keep[-1].append(i)
                            break
            counter -= 1
            if not indices_to_keep[-1]:
                break
        
        # convert back to single list of indices to keep
        indices_to_keep = [i for l in indices_to_keep for i in l]
        # return list of indices we now want to remove
        return np.setdiff1d(range(len(self.pops)), indices_to_keep)
    
    @staticmethod
    def is_hermitian(A):
        '''Tests Hermiticity of numpy array A. See https://en.wikipedia.org/wiki/Hermitian_matrix'''
        return A.shape[0] == A.shape[1] and np.allclose(A, A.conj().T)
        
    @staticmethod
    def is_dissipator_physical(D, pops):
        '''Tests if Lindblad dissipator is physical ie. columns corresponding to populations sum to zero.'''
        return np.allclose(np.dot(D.T, pops), np.zeros(pops.size))