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
    
    def test_noise_single_freq1(self):
        Gamma_R = 1.
        Gamma_L = 1.
        expected_F2 = utils.second_order_fano_factor_srl(Gamma_L, Gamma_R, 0)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
        
    def test_noise_single_freq2(self):
        Gamma_R = 0.5
        Gamma_L = 1.
        expected_F2 = utils.second_order_fano_factor_srl(Gamma_L, Gamma_R, 0)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        self.assertAlmostEqual(expected_F2, srl_solver.second_order_fano_factor(0))
    
    def test_noise_multiple_freqs(self):
        Gamma_R = 1.
        Gamma_L = 1.
        freq = np.linspace(0,10,1000)
        expected_F2 = utils.second_order_fano_factor_srl(Gamma_L, Gamma_R, freq)
        srl_solver = utils.setup_srl_solver(Gamma_L, Gamma_R)
        npt.assert_allclose(expected_F2, srl_solver.second_order_fano_factor(freq), rtol=1.e-3)
        
        
        