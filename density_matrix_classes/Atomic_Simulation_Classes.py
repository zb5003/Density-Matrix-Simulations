import time
from density_matrix_classes.nlevel import *


def sigmoid(t, tau, tau_0):
    """
    Represents the turn on of a beam.
    :param t: Float. The time.
    :param tau: Float. Characteristic time scale for the turn on of the beam.
    :param tau_0: Float. Time at which the beam turns on.
    :return: Float. Amplitude of the beam as it turns on.
    """
    return 1 / (1 + sp.exp(-(t - tau_0) / tau))

class atom:

    def __init__(self,
                 state,
                 decay,
                 closed):
        """
        Create an atom instance with initial state 'state' whose levels decay according to the 
        matrix decay.
        :param state: The initial state of the system.  This is used in two attributes: one to 
                      save the initial state (self.initial_state) and one to update as the
                      atom evolves (self.current_state).
        :param decay: Matrix describing the decay of the system.
        :param closed: Array detailing the decay rate from each excited state to each ground state.
                       This array is used to add population back into the system, thus making it closed.
                       The (i, j)th element represents decay from the jth state to the ith state.
                       The elements of the array should be the square roots of the transition decay rates.
        """
        self.initial_state = state
        self.current_state = state
        self.decay = decay
        self.closed = closed
        self.c, self.c_T = self.generate_c(closed)

    def generate_c(self, dec_to_mat):
        """
        Splits dec_to_mat into two lists.  One list is a list of lower operators, the other is a list of raising operators.
        See documentation for more detail.
        :param dec_to_mat: Array. Describes the decay from state j to state i.
        :return: Two lists. The first contains lower operators relevant to the decay, the other contains the relevant raising operators.
        """
        dim1, dim2 = sp.shape(dec_to_mat)
        dim3 = sp.count_nonzero(dec_to_mat)
        c = sp.zeros((dim3, dim1, dim2))
        c_T = sp.zeros((dim3, dim1, dim2))
        k = 0
        for i in range(dim1 - 1):
            for j in range(i + 1, dim1):
                if dec_to_mat[i, j] != 0:
                    c[k, i, j] = dec_to_mat[i, j]
                    c_T[k, j, i] = dec_to_mat[i, j]
                    k = k + 1
        # print(c, c_T)
        return c, c_T

    def evolve_step(self,
                    hamiltonian,
                    dt):
        """
        Preform one step of the time evolution of the atom using the Runge-Kutta
        fourth order method (found in nlevel.py).
        The instance attribute current_state is updated to the new time-evolved state.
        :param hamiltonian: Array. The Hamiltonian at a particular time.
        :param dt: Float. Time step over which to preform the time evolution.
        :return: Complex array. The time-evolved state.
        """
        self.current_state = RK_rho(hamiltonian,
                                    self.decay,
                                    self.current_state,
                                    [self.c, self.c_T],
                                    dt)
        return self.current_state

class hamiltonian_construct:

    def __init__(self,
                 dipoles,
                 field,
                 freq,
                 pulse_sequence=sigmoid,
                 pulse_params=(9e-9, 90e-9)):
        """
        This class produces a realistic time dependent Hamiltonian (i.e. includes laser turn on) 
        in the interaction picture and dipole approximation. Each instance describes a single beam.
        :param dipoles: Array. Matrix of dipole moments between states (could be electric or magnetic).
        :param field: Float. Amplitude of the (electric or magnetic) field.
        :param freq: Array. Matrix containing the frequencies at which the Hamiltonian matrix elements
                     oscillate.
        :param pulse_sequence: Function.  Describes the pulse sequence of the beam.
        :param pulse_params: Tuple.  Contains the parameters, other than time, for pulse_sequence.
        """
        self.dipoles = dipoles
        self.field = field
        self.freq = freq
        self.pulse_sequence = pulse_sequence
        self.pulse_params = pulse_params

        # For reference
        self.rabi_freqs = dipoles * field / hbar

    def carrier(self, t):
        """
        The interaction picture Hamiltonian with instantaneous laser turn on time. 
        :param t: Float. The time.
        :return: Complex array. The constant amplitude Hamiltonian at time t.
        """
        time_dep = sp.exp(1j * self.freq * t)
        return self.field * sp.multiply(self.dipoles, time_dep)

    def envelope(self, t):
        """
        Caluclates the modulation of the constant amplitude Hamiltonian by the pulse sequence.
        The output needs to be multiplied by the full filed amplitude before being used in the
        Hamiltonian.
        :param t: Float. The time.
        :return: Float. Relative field amplitude at time t.
        """
        all_params = (t, ) + self.pulse_params
        return self.pulse_sequence(*all_params)

    def hamiltonian(self, t):
        """
        Constructs the physical Hamiltonian (in the interaction picture)
         at time t by combining the results of envelope() and carrier().
        :param t: Float. The time.
        :return: Complex array. The full Hamiltonian at time t.
        """
        return - self.envelope(t) * self.carrier(t) / 2

class single_atom_simulation:

    def __init__(self,
                 system,
                 ham_obj,
                 nt,
                 dt):
        """
        Create an instance of a simulation that calculates the time evolution of a single atom interacting
        with a single laser.
        :param system: An instance of the class atom.
        :param ham_obj: A list of instances of the class hamiltonian_construct.
                        IMPORTANT: The first element of this list will be the laser that is
                                   swept for the susceptibility plot.
        :param nt: Float. Number of time steps in the simulation.
        :param dt: Float. Time step.
        """
        self.system = system
        self.ham_obj = ham_obj
        self.nt = nt
        self.dt = dt

        self.duration = nt * dt
        self.evolver = lambda t: sum([k.hamiltonian(t) for k in ham_obj])
        self.freq_default = ham_obj[0].freq.copy()
        self.mask = self.generate_mask()

    def generate_mask(self):
        """
        Records the nonzero matrix elements of the Hamiltonian.  The mask will mainly be used for detuning the laser beam.
        :return: Array. Elements are one for nonzero Hamiltonian elements and zero otherwise.
        """
        detune_mask = sp.zeros(sp.shape(self.system.initial_state))  # Set / reset mask
        detune_mask[self.ham_obj[0].dipoles != 0] = 1  # Select only nonzero matrix elements of the Hamiltonian
        return sp.triu(detune_mask) - sp.tril(detune_mask)

    def detune(self, detuning, ham=0):
        """
        Detunes the frequencies in the Hamiltonian specified by ham.
        :param detuning: Float.  By how much the beam will be detuned by. Negative for red detuning, positive for blue.
        :param ham: Specify which beam to detune if there are multiple beams.  The default is the beam that comes first in ham_obj.
        :return: None.
        """
        self.ham_obj[ham].freq = self.freq_default + self.mask * detuning
        return None

    def reset_state(self):
        """
        Return the system to its original state.
        :return: None.
        """
        self.system.current_state = self.system.initial_state
        return None

    def envelope_extract(self, state, t):
        """
        Extract the envelope of the density matrix amplitudes.
        :param state: Complex array. The density matrix.
        :param t: Float. The time.
        :return: Complex array. Matrix containing the amplitudes of envelope of the density matrix elements.
        """
        return state * sp.exp(-1j * self.ham_obj[0].freq * t)

    def final_state(self):
        """
        Calculate the state of the system after nt timesteps of size dt.
        :return: Complex array. The final density matrix of the system.
        """
        times = sp.linspace(0, self.duration, self.nt, endpoint=False)
        for i in times:
            self.system.evolve_step(self.evolver(i), self.dt)
        final_untrans = self.envelope_extract(self.system.current_state, (self.nt - 1) * self.dt)

        return final_untrans

    def susceptibility(self, detunings):
        """
        Calculate the frequency dependent susceptibility for a given system interacting with an arbitrary number of lasers,
        potentially each with different dipole moments.
        :param detunings:  Array. The detunings (in Hz) to sweep through for the susceptibility plot.
        :return: Complex array. The value of the susceptibility at each detuning.
        """
        dim1, dim2 = sp.shape(self.system.initial_state)
        chi = sp.zeros((len(detunings), dim1, dim2), dtype=complex)

        for index, i in enumerate(detunings):
            self.detune(i)
            chi[index] = self.final_state()
            self.reset_state()

        return chi

    def time_evolution(self, detuning=0):
        """
        Calculate the state of the system at each of the nt time steps of size dt.
        :return: Complex arrat. The density matrix at ever timestep of the simulation.
        """
        t1 = time.time()
        self.detune(detuning)
        row, column = sp.shape(self.system.initial_state)
        times = sp.linspace(0, self.duration, self.nt, endpoint=False)
        time_dep_state = sp.zeros((self.nt, row, column), dtype=complex)

        for index, i in enumerate(times):
            time_dep_state[index] = self.envelope_extract(self.system.evolve_step(self.evolver(i), self.dt), i)
        t2 = time.time() - t1
        print("Detuning =", round(detuning / 1e6, 4), "MHz", "Time elapsed =", str(round(t2, 4)), "seconds")

        return time_dep_state
