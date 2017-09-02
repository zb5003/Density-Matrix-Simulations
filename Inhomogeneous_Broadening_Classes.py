import scipy as sp
import time
from Atomic_Simulation_Classes import *

class inhomogeneous_broadening:

    def __init__(self,
                 sing_sim,
                 linewidth,
                 n_atoms):
        """

        :param sing_sim
        :param linewidth:
        :param n_atoms:
        """
        self.sing_sim = sing_sim
        self.linewidth = linewidth
        self.n_atoms = n_atoms

        self.detunings = sp.linspace(-linewidth / 2, linewidth / 2, n_atoms)

    def broadened_time_evolution(self):
        """
        Calculate the state of all the atoms in the inhomogeneous line at each of nt time steps of size dt.
        :return: The state of the system at each timestep averaged over the inhomogeneous line.
        """
        dim1, dim2 = sp.shape(self.sing_sim.system.initial_state)
        times = sp.linspace(0, self.sing_sim.duration, self.sing_sim.nt, endpoint=False)
        time_dep_state = sp.zeros((self.sing_sim.nt, dim1, dim2), dtype=complex)
        for index_i, i in enumerate(self.detunings):
            self.sing_sim.reset_state()
            self.detune(i)
            t1 = time.time()
            for index_j, j in enumerate(times):
                time_dep_state[index_j] = time_dep_state[index_j] \
                                      + self.sing_sim.system.evolve_step(self.sing_sim.evolver(j), self.sing_sim.dt).copy() * \
                                        sp.exp(-1j * (self.sing_sim.ham_obj[0].freq) * j)
            print("Atom number =",index_i, "Detuning =", round(i / 1e6, 4), "MHz",
                  "Time elapsed =", str(round(time.time() - t1, 4)), "seconds")

        return time_dep_state / self.n_atoms

    def detune(self,
               detuning):
        """
        Detune the original Hamiltonian by detuning.
        :param detuning: The detuning in Hz.
        :return: None.
        """
        self.sing_sim.ham_obj[0].freq = self.sing_sim.freq_default + self.sing_sim.mask * detuning

        return None
