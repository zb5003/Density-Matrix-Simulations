import scipy as sp
from Atomic_Simulation_Classes import *

class inhomogeneous_broadening:

    def __init__(self, sing_sim, linewidth, n_atoms):
        """

        :param sing_sim
        :param linewidth:
        :param n_atoms:
        """
        self.sing_sim = sing_sim
        self.linewidth = linewidth
        self.n_atoms = n_atoms

        self.detunings = self.generate_detunings

    def generate_detunings(self):
        """
        Generate the detunings of the atoms due to the inhomogeneous broadening.
        :return: An array containing the detunings.
        """
        return sp.linspace(-self.linewidth / 2, self.linewidth / 2, self.n_atoms)

    def detune(self, detuning):
        """
        Detune the original Hamiltonian by detuning.
        :param detuning: The detuning in Hz.
        :return: None.
        """
        self.sing_sim.reset_detuning()
        self.sing_sim.ham_obj[0].freq = self.sing_sim.mask * detuning

        return None

    def broadened_time_evolution(self):
        """

        :return:
        """