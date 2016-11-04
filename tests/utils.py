import numpy as np
from counting_statistics.lindblad_system import LindbladSystem

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
                  [0, -bias/2., T_c],
                  [0, T_c, bias/2.]])
    ops = np.array([np.array([[0, 0, 0],[1., 0, 0],[0, 0, 0]]), np.array([[0, 0, 1.],[0, 0, 0],[0, 0, 0]])])
    rates = np.array([Gamma_L, Gamma_R])
    return LindbladSystem(H, ops, rates, reduce_dim=reduce_dim)

def reduced_srl_liouvillian(Gamma_L, Gamma_R):
    return np.array([[-Gamma_L, Gamma_R],
                     [Gamma_L, -Gamma_R]])
    
def reduced_dqd_liouvillian(bias, T_c, Gamma_L, Gamma_R):
    return np.array([[-Gamma_L, 0, 0, 0, Gamma_R],
                     [Gamma_L, 0, T_c*1.j, -T_c*1.j, 0],
                     [0, T_c*1.j, -Gamma_R/2.+bias*1.j, 0, -T_c*1.j],
                     [0, -T_c*1.j, 0, -Gamma_R/2.-bias*1.j, T_c*1.j],
                     [0, 0, -T_c*1.j, T_c*1.j, -Gamma_R]])