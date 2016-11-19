import unittest
import numpy as np
import numpy.testing as npt
from tests import utils
from counting_statistics.fcs_solver import FCSSolver
# maybe want to use numpy test suite

class FCSSolverTestCase(unittest.TestCase):
    
    def test_stationary_state(self):
        pass
    
    def test_mean_SRL(self):
        Gamma_L = 1.
        Gamma_R = 1.
        expected_mean = utils.mean_srl(Gamma_L, Gamma_R)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_mean, srl_solver.mean())
    
    def test_mean_DQD(self):
        pass
    
    def test_zero_freq_F2_1(self):
        Gamma_R = 1.
        Gamma_L = 1.
        expected_F2 = utils.second_order_fano_factor_srl(Gamma_L, Gamma_R, 0)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
        
    def test_zero_freq_F2_2(self):
        Gamma_R = 0.5
        Gamma_L = 1.
        expected_F2 = utils.second_order_fano_factor_srl(Gamma_L, Gamma_R, 0)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
    
    def test_F2_multiple_freqs(self):
        Gamma_R = 1.
        Gamma_L = 1.
        freq = np.linspace(0,10,1000)
        expected_F2 = utils.second_order_fano_factor_srl(Gamma_L, Gamma_R, freq)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        npt.assert_allclose(expected_F2, srl_solver.second_order_fano_factor(freq), rtol=1.e-3)
    
    def test_setattr(self):
        '''Testing overidden __setattr__ function which changes cache_is_stale flag.'''
        srl_solver = utils.setup_srl_solver(1., 1.)
        self.assertTrue(srl_solver._FCSSolver__cache_is_stale) # check __cache_is_stale is True on initialization
        
        # change of variable in srl_solver.__watch_variables should trigger stale cache
        srl_solver._FCSSolver__cache_is_stale = False
        self.assertFalse(srl_solver._FCSSolver__cache_is_stale)
        srl_solver.H = np.array([[1.,0],[0,1.]])
        self.assertTrue(srl_solver._FCSSolver__cache_is_stale)
        
        srl_solver._FCSSolver__cache_is_stale = False
        self.assertFalse(srl_solver._FCSSolver__cache_is_stale)
        srl_solver.D_ops = [np.array([[0,1],[0,0]]), np.array([[0,0],[1,0]])]
        self.assertTrue(srl_solver._FCSSolver__cache_is_stale)
        
        srl_solver._FCSSolver__cache_is_stale = False
        self.assertFalse(srl_solver._FCSSolver__cache_is_stale)
        srl_solver.D_rates = [2., 2.]
        self.assertTrue(srl_solver._FCSSolver__cache_is_stale)
        
        srl_solver._FCSSolver__cache_is_stale = False
        self.assertFalse(srl_solver._FCSSolver__cache_is_stale)
        srl_solver.jump_idx = np.array([0, 1])
        self.assertTrue(srl_solver._FCSSolver__cache_is_stale)
        
        srl_solver._FCSSolver__cache_is_stale = False
        self.assertFalse(srl_solver._FCSSolver__cache_is_stale)
        srl_solver.reduce_dim = True
        self.assertTrue(srl_solver._FCSSolver__cache_is_stale)
        
        # variables not in srl_solver.__watch_variables shouldn't trigger stale cache
        srl_solver._FCSSolver__cache_is_stale = False
        self.assertFalse(srl_solver._FCSSolver__cache_is_stale)
        srl_solver.sys_dim = 8
        self.assertFalse(srl_solver._FCSSolver__cache_is_stale)
        
    def test_cache_is_stale_after_refresh(self):
        srl_solver = utils.setup_srl_solver(1., 1.)
        self.assertTrue(srl_solver._FCSSolver__cache_is_stale) # check __cache_is_stale is True on initialization
        
        # cache should not be stale after refresh
        srl_solver.refresh_cache()
        self.assertFalse(srl_solver._FCSSolver__cache_is_stale)
        
    def test_cache_refresh_D_rates(self):
        '''Functionality test for refresh cache behaviour.'''
        Gamma_L = 1.
        Gamma_R = 1.
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        expected_F2 = utils.second_order_fano_factor_srl(Gamma_L, Gamma_R, 0)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
        
        Gamma_R = 0.5
        srl_solver.D_rates = np.array([Gamma_R, Gamma_L])
        expected_F2 = utils.second_order_fano_factor_srl(Gamma_L, Gamma_R, 0)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
        
    
    
