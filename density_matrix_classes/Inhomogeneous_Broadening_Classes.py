import scipy as sp
import time
from density_matrix_classes.Atomic_Simulation_Classes import *

class inhomogeneous_broadening:

    def __init__(self,
                 sing_sim,
                 linewidth,
                 n_atoms):
        """

        :param sing_sim: Instance of single_atom_simulation.
        :param linewidth: Inhomogeneous linewidth.
        :param n_atoms: Number of atoms to be spread throughout the inhomogeneous linewidth.
        """
        self.sing_sim = sing_sim
        self.linewidth = linewidth
        self.n_atoms = n_atoms

        self.detunings = sp.linspace(-linewidth / 2, linewidth / 2, n_atoms)

    def broadened_time_evolution(self):
        """
        -Calculate the state of all the atoms in the inhomogeneous line at each of nt time steps of size dt.
        -Runs in serial using time_evolve_serial() from the single_atom_simulation class.
        -The detuning of each atom is produced by using the detune() function of this class, not the detuning arg in
         time_evolve_serial()
        :return: The state of the system at each timestep averaged over the inhomogeneous line.
        """
        dim1, dim2 = sp.shape(self.sing_sim.system.initial_state)
        time_dep_state = sp.zeros((self.sing_sim.nt, dim1, dim2), dtype=complex)
        for index_i, i in enumerate(self.detunings):
            self.sing_sim.reset_state()
            self.detune(i)
            t1 = time.time()
            time_dep_state = time_dep_state + self.sing_sim.time_evolution_serial()
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

    def broadened_susceptibility(self):
        """
        
        :return: 
        """