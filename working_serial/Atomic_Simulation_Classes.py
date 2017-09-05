from nlevel import *

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
                       The elements of the array should be the square roots of the decay rates.
        """
        self.initial_state = state
        self.current_state = state
        self.decay = decay
        self.closed = closed
        self.c, self.c_T = self.generate_c(closed)

    def generate_c(self, dec_to_mat):
        """
        
        :param dec_to_mat: 
        :return: 
        """
        dim1, dim2 = sp.shape(dec_to_mat)
        dim3 = sp.count_nonzero(dec_to_mat)
        c = sp.zeros((dim3, dim1, dim2))
        c_T = sp.zeros((dim3, dim1, dim2))
        for k in range(dim3):
            for i in range(dim1 - 1):
                for j in range(i + 1, dim1):
                    if dec_to_mat[i, j] != 0:
                        c[k, i, j] = dec_to_mat[i, j]
                        c_T[k, j, i] = dec_to_mat[i, j]
        return c, c_T

    def evolve_step(self,
                    hamiltonian,
                    dt):
        """
        Preform one step of the time evolution of the atom using the Runge-Kutta
        fourth order method (found in nlevel.py).
        The instance attribute current_state is updated to the new time-evolved state.
        :param hamiltonian: The Hamiltonian at a particular time.
        :param dt: Time step over which to prefomr the time evolution.
        :return: The time-evolved state.
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
                 freq):
        """
        This class produces a realistic time dependent Hamiltonian (i.e. includes laser turn on) 
        in the interaction picture and dipole approximation.
        :param dipoles: Matrix of dipole moments between states (could be electric or magnetic).
        :param field: Amplitude of the (electric or magnetic) field.
        :param freq: Matrix containing the frequencies at which the Hamiltonian matrix elements
                     oscillate.
        """
        self.dipoles = dipoles
        self.field = field
        self.freq = freq

        # For reference
        self.rabi_freqs = dipoles * field / hbar

        # Turn on parameters for a ~80 MHz AOM.
        self.tau = 9e-9
        self.tau_0 = 90e-9

    def carrier(self, t):
        """
        The interaction picture Hamiltonian with instantaneous laser turn on time. 
        :param t: Time.
        :return: The Hamiltonian at time t.
        """
        time_dep = sp.exp(1j * self.freq * t)
        return self.field * sp.multiply(self.dipoles, time_dep)

    def envelope(self, t):
        """
        A sigmoid function used to model the turn on time of the laser.
        The output needs to be multiplied by the full filed amplitude before being used in the
        Hamiltonian.
        :param t: Time.
        :return: Relative field amplitude at time t.
        """
        return 1 / (1 + sp.exp(-(t - self.tau_0) / self.tau))

    def hamiltonian(self, t):
        """
        Construct the physical Hamiltonian (in the interaction picture)
         at time t which includes the laser turn on time.
        :param t: Time.
        :return: The physical Hamiltonian at time t.
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
        with lasers at fixed frequencies.
        :param system: An instance of the class atom.
        :param ham_obj: A list of instances of the class hamiltonian_construct.
                        IMPORTANT: The first element of this list will be the laser that is
                                   swept for the susceptibility plot.
        :param nt: Number of time steps in the simulation.
        :param dt: Time step.
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

        :return:
        """
        detune_mask = sp.zeros(sp.shape(self.system.initial_state))  # Set / reset mask
        detune_mask[self.ham_obj[0].dipoles != 0] = 1  # Select only nonzero matrix elements of the Hamiltonian
        return sp.triu(detune_mask) - sp.tril(detune_mask)

    def detune(self, detuning):
        """
        
        :param detuning: 
        :return: 
        """
        self.ham_obj[0].freq = self.freq_default + self.mask * detuning
        return None

    def final_state(self):
        """
        Calculate the state of the system after nt timesteps of size dt.
        :return: The final state of the system (matrix).
        """
        times = sp.linspace(0, self.duration, self.nt, endpoint=False)
        for i in times:
            self.system.evolve_step(self.evolver(i), self.dt)
        final_untrans = self.system.current_state * \
                        sp.exp(-1j * self.ham_obj[0].freq * (self.nt - 1) * self.dt)

        return final_untrans

    def reset_state(self):
        """
        Return the system to its original state.
        :return: None.
        """
        self.system.current_state = self.system.initial_state
        return None

    def susceptibility(self, detunings):
        """
        Calculate the susceptibility curve for a given system interacting with an arbitrary number of lasers,
        potentially each with different dipole moments.

        Note that the susceptibility is defined for a single frequency.
        :param detunings:  The detunings (in Hz) to sweep through for the susceptibility plot.
        :return: The susceptibility plot.
        """
        dim1, dim2 = sp.shape(self.system.initial_state)
        chi = sp.zeros((len(detunings), dim1, dim2), dtype=complex)

        for index, i in enumerate(detunings):
            self.detune()
            chi[index] = self.final_state()
            self.reset_state()

        return chi

    def time_evolution(self):
        """
        Calculate the state of the system at each of the nt time steps of size dt.
        :return: The state at ever timestep of the simulation.
        """

        row, column = sp.shape(self.system.initial_state)
        times = sp.linspace(0, self.duration, self.nt, endpoint=False)
        time_dep_state = sp.zeros((self.nt, row, column), dtype=complex)

        for index, i in enumerate(times):
            time_dep_state[index] = self.system.evolve_step(self.evolver(i),
                                                            self.dt).copy() * \
                                               sp.exp(-1j * self.ham_obj[0].freq * i)

        return time_dep_state
