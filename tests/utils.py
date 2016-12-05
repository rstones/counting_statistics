import numpy as np
from counting_statistics.lindblad_system import LindbladSystem
from counting_statistics.fcs_solver import FCSSolver

def setup_srl_system(Gamma_L, Gamma_R, reduce_dim):
    '''Instantiates LindbladSystem for single resonant level model'''
    H = np.array([[0,0],
                  [0,0]])
    ops = np.array([np.array([[0, 0],[1, 0]]), np.array([[0, 1],[0, 0]])])
    rates = np.array([Gamma_L, Gamma_R])
    return LindbladSystem(H, ops, rates, reduce_dim=reduce_dim)

def setup_dqd_system(bias, T_c, Gamma_L, Gamma_R, reduce_dim):
    '''Instantiates LindbladSystem for double quantum dot model'''
    H = np.array([[0, 0, 0],
                  [0, bias/2., T_c],
                  [0, T_c, -bias/2.]])
    ops = np.array([np.array([[0, 0, 0],[1., 0, 0],[0, 0, 0]]), np.array([[0, 0, 1.],[0, 0, 0],[0, 0, 0]])])
    rates = np.array([Gamma_L, Gamma_R])
    return LindbladSystem(H, ops, rates, reduce_dim=reduce_dim)

def reduced_srl_liouvillian(Gamma_L, Gamma_R):
    return np.array([[-Gamma_L, Gamma_R],
                     [Gamma_L, -Gamma_R]])
    
def reduced_dqd_liouvillian(bias, Tc, Gamma_L, Gamma_R):
    return np.array([[-Gamma_L, 0, 0, 0, Gamma_R],
                     [Gamma_L, 0, Tc*1.j, -Tc*1.j, 0],
                     [0, Tc*1.j, -Gamma_R/2.-bias*1.j, 0, -Tc*1.j],
                     [0, -Tc*1.j, 0, -Gamma_R/2.+bias*1.j, Tc*1.j],
                     [0, 0, -Tc*1.j, Tc*1.j, -Gamma_R]])
    
def mean_srl(Gamma_L, Gamma_R):
    return Gamma_L*Gamma_R / (Gamma_L + Gamma_R)
    
def zero_freq_F2_srl(Gamma_L, Gamma_R):
    return (Gamma_L**2 + Gamma_R**2) / (Gamma_L+Gamma_R)**2

def setup_srl_solver(Gamma_L, Gamma_R):
    return FCSSolver(reduced_srl_liouvillian(Gamma_L, Gamma_R), np.array([[0,1],[0,0]]), np.array([1,1]))

def setup_srl_solver_from_hilbert_space(Gamma_L, Gamma_R):
    return FCSSolver.from_hilbert_space(np.array([[0,0],[0,0]]), [np.array([[0,1],[0,0]]), np.array([[0,0],[1,0]])], \
                                [Gamma_R, Gamma_L], np.array([1,0]), reduce_dim=True)
    
def mean_dqd(Gamma_L, Gamma_R, Tc, bias):
    return (4. * Tc**2 * Gamma_R * Gamma_L) / (4.*Gamma_L*bias**2 + 4.*Tc**2*Gamma_R + Gamma_L*(8.*Tc**2 + Gamma_R**2))
    
def zero_freq_F2_dqd(Gamma_L, Gamma_R, Tc, bias):
    return (16.*(4.*Gamma_L**2 + Gamma_R**2)*Tc**4 + 8.*Gamma_L**2 * (12.*bias**2 - Gamma_R**2)*Tc**2 + Gamma_L**2 * (4.*bias**2 + Gamma_R**2)**2) \
                / (4.*Gamma_L*bias**2 + Gamma_L*Gamma_R**2 + 4.*Tc**2 * (2.*Gamma_L + Gamma_R))**2
                
def setup_dqd_solver(Gamma_L, Gamma_R, Tc, bias):
    L = reduced_dqd_liouvillian(bias, Tc, Gamma_L, Gamma_R)
    jump_op = np.zeros((5,5))
    jump_op[0,4] = 1
    return FCSSolver(L, jump_op, np.array([1,1,1,0,0]))
    
def setup_dqd_solver_from_hilbert_space(Gamma_L, Gamma_R, Tc, bias):
    return FCSSolver.from_hilbert_space(np.array([[0,0,0],[0,bias/2.,Tc],[0,Tc,-bias/2.]]), \
                     [np.array([[0,0,1.],[0,0,0],[0,0,0]]), np.array([[0,0,0],[1.,0,0],[0,0,0]])], \
                     [Gamma_R, Gamma_L], np.array([1,0]), reduce_dim=True)
    
def finite_freq_F2(freq, liouvillian, jump_op):
    '''Calculation from the analytic expression for finite frequency second order Fano factor of systems with a single element
    in the jump operator.'''
    evals,evecs = np.linalg.eig(liouvillian)
    c = np.dot(np.linalg.inv(evecs), np.dot(jump_op, evecs))
    F2 = 1.
    for i in range(liouvillian.shape[0]):
        if np.abs(evals[i]) < 1.e-7: continue # don't include zero eigenvalues in sum
        F2 -= 2. *  (c[i,i]*evals[i] / (freq**2 + evals[i]**2))
    return np.real(F2)

def skewness_srl(freq1, freq2, Gamma_L, Gamma_R):
    F3 = np.zeros((freq1.size, freq2.size))
    for i in range(freq1.size):
        for j in range(freq2.size):
            F3[i,j] = 1. - 2. * Gamma_L * Gamma_R * \
                                (((Gamma_L**2 + Gamma_R**2 + freq1[i]**2 - freq1[i]*freq2[j] + freq2[j]**2) \
                                  * (3.*(Gamma_L + Gamma_R)**2 + freq1[i]**2 - freq1[i]*freq2[j] + freq2[j]**2)) \
                                 / (((Gamma_L + Gamma_R)**2 + freq1[i]**2) \
                                    * ((Gamma_L + Gamma_R)**2 + freq2[j]**2) \
                                    * ((Gamma_L + Gamma_R)**2 + (freq1[i]-freq2[j])**2)))
    return F3
