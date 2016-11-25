import unittest
import numpy as np
import numpy.testing as npt
from tests import utils
from counting_statistics.fcs_solver import FCSSolver

class FCSSolverTestCase(unittest.TestCase):
    
    def test_stationary_state(self):
        pass
    
    def test_mean_srl(self):
        '''Test of mean calculation for SRL model.'''
        Gamma_L = 1.; Gamma_R = 0.5
        expected_mean = utils.mean_srl(Gamma_L, Gamma_R)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_mean, srl_solver.mean())
    
    def test_mean_dqd(self):
        '''Test of mean calculation for DQD model.'''
        Gamma_L = 1.; Gamma_R = 0.5; Tc = 3.; bias = 1.
        expected_mean = utils.mean_dqd(Gamma_L, Gamma_R, Tc, bias)
        dqd_solver = utils.setup_dqd_solver(Gamma_L, Gamma_R, Tc, bias)
        self.assertAlmostEqual(expected_mean, dqd_solver.mean())
    
    def test_zero_freq_F2_srl(self):
        '''Functionality test for second order Fano factor of SRL model with single frequency.'''
        Gamma_R = 0.5; Gamma_L = 1.
        expected_F2 = utils.zero_freq_F2_srl(Gamma_L, Gamma_R)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
        
    def test_zero_freq_F2_dqd(self):
        '''Functionality test for second order Fano factor of DQD model with single frequency.'''
        Gamma_L = 1.; Gamma_R = 0.5; Tc = 3.; bias = 1.
        expected_F2 = utils.zero_freq_F2_dqd(Gamma_L, Gamma_R, Tc, bias)
        dqd_solver = utils.setup_dqd_solver(Gamma_L, Gamma_R, Tc, bias)
        self.assertAlmostEqual(expected_F2, dqd_solver.second_order_fano_factor(0))
    
    def test_finite_freq_F2_srl(self):
        '''Functionality test for second order Fano factor using SRL model with an array of frequencies.'''
        Gamma_R = 0.5; Gamma_L = 1.
        freq = np.linspace(0,10,100)
        srl_liouvillian = utils.reduced_srl_liouvillian(Gamma_L, Gamma_R)
        srl_jump_op = np.array([[0, Gamma_R],[0, 0]])
        expected_F2 = utils.finite_freq_F2(freq, srl_liouvillian, srl_jump_op)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        npt.assert_allclose(expected_F2, srl_solver.second_order_fano_factor(freq))
    
    def test_finite_freq_F2_dqd(self):
        '''Functionality test for second order Fano factor using DQD model with an array of frequencies.'''
        Gamma_L = 1.; Gamma_R = 0.5; Tc = 3.; bias = 1.
        freq = np.linspace(0,10,100)
        dqd_liouvillian = utils.reduced_dqd_liouvillian(bias, Tc, Gamma_L, Gamma_R)
        dqd_jump_op = np.zeros((5,5))
        dqd_jump_op[0,4] = Gamma_R
        expected_F2 = utils.finite_freq_F2(freq, dqd_liouvillian, dqd_jump_op)
        dqd_solver = utils.setup_dqd_solver(Gamma_L, Gamma_R, Tc, bias)        
        npt.assert_allclose(expected_F2, dqd_solver.second_order_fano_factor(freq))
        
    def test_finite_freq_F3_srl(self):
        Gamma_L = 1.; Gamma_R = 1.
        freq1 = np.linspace(1,5,10)
        freq2 = np.linspace(6,10,10)
        expected_F3 = utils.skewness_srl(freq1, freq2, Gamma_L, Gamma_R)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        npt.assert_allclose(expected_F3, srl_solver.third_order_fano_factor(freq1, freq2))
    
    def test_setattr(self):
        '''Testing overidden __setattr__ function in FCSSolver which changes __cache_is_stale flag.'''
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
        expected_F2 = utils.zero_freq_F2_srl(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
        
        Gamma_R = 0.5
        srl_solver.D_rates = np.array([Gamma_R, Gamma_L])
        expected_F2 = utils.zero_freq_F2_srl(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
        
    
    
