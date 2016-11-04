import unittest
import numpy as np
import numpy.testing as npt
from counting_statistics.lindblad_system import LindbladSystem
from tests import utils

class LindbladSystemTestCase(unittest.TestCase):
    
    def test_real_H_hermitian(self):
        H = np.array([[0, 1.],[1., 0]])
        self.assertTrue(LindbladSystem.is_hermitian(H))
        
    def test_real_H_not_hermitian(self):
        H = np.array([[0, 1.],[2., 0]])
        self.assertFalse(LindbladSystem.is_hermitian(H))
    
    def test_complex_H_hermitian(self):
        H = np.array([[0, 1.j],[-1.j, 0]])
        self.assertTrue(LindbladSystem.is_hermitian(H))
    
    def test_complex_H_not_hermitian(self):
        H = np.array([[0, 1.j],[1.j, 0]])
        self.assertFalse(LindbladSystem.is_hermitian(H))
        
    def test_non_square_H_not_hermitian(self):
        H = np.array([[0, 1., 0],[1., 0, 0]])
        self.assertFalse(LindbladSystem.is_hermitian(H))
    
    def test_dissipator_physical(self):
        D = np.array([[1., -1.],[-1., 1.]])
        pops = np.array([1, 1])
        self.assertTrue(LindbladSystem.is_dissipator_physical(D, pops))
    
    def test_dissipator_not_physical(self):
        D = np.array([[1., 1.],[1., 1.]])
        pops = np.array([1, 1])
        self.assertFalse(LindbladSystem.is_dissipator_physical(D, pops))
        
    def test_ops_and_rates_different_length_raises_error(self):
        H = np.array([[0, 1.j],[-1.j, 0]])
        ops = np.array([np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])])
        rates = np.array([1,2,3])
        self.failUnlessRaises(ValueError, LindbladSystem, H, ops, rates)
        
    def test_reduce_dim_SRL(self):
        '''Functionality test of dimension reduction of Liouvillian for a two level system'''
        Gamma_L = 1.
        Gamma_R = 2.
        ls = utils.setup_srl_system(Gamma_L, Gamma_R, True)
        npt.assert_allclose(ls.liouvillian(), utils.reduced_srl_liouvillian(Gamma_L, Gamma_R))
    
    def test_reduce_dim_DQD(self):
        '''Functionality test of dimension reduction of Liouvillian for a double quantum dot system'''
        bias = 1.
        T_c = 3.
        Gamma_L = 1.
        Gamma_R = 2.
        ls = utils.setup_dqd_system(bias, T_c, Gamma_L, Gamma_R, True)
        npt.assert_allclose(ls.liouvillian(), utils.reduced_dqd_liouvillian(bias, T_c, Gamma_L, Gamma_R))