from nlevel import *

class atom:

    def __init__(self, state, decay):
        """
        Create an atom instance with initial state 'state' whose levels decay according to the 
        matrix decay.
        :param state: The initial state of the system.  This is used in two attributes: one to 
                      save the initial state (self.initial_state) and one to update as the
                      atom evolves (self.current_state).
        :param decay: Matrix describing the decay of the system.
        """
        self.initial_state = state
        self.current_state = state
        self.decay = decay

    def evolve_step(self, hamiltonian, dt):
        """
        Preform one step of the time evolution of the atom using the Runge-Kutta
        fourth order method (found in nlevel.py).
        The instance attribute current_state is updated to the new time-evolved state.
        :param hamiltonian: The Hamiltonian at a particular time.
        :param dt: Time step over which to prefomr the time evolution.
        :return: The time-evolved state.
        """
        self.current_state = RK_rho(hamiltonian, self.decay, self.current_state, dt)
        return self.current_state

class hamiltonian_construct:

    # Tunr on parameters for a ~80 MHz AOM.
    tau = 9e-9
    tau_naught = 90e-9

    def __init__(self, dipoles, field, freq):
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

        self.rabi_freqs = dipoles * field / hbar  # for reference

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
        return 1 / (1 + sp.exp(-(t - hamiltonian_construct.tau_naught) / hamiltonian_construct.tau))

    def hamiltonian(self, t):
        """
        Construct the physical Hamiltonian (in the interaction picture)
         at time t which includes the laser turn on time.
        :param t: Time.
        :return: The physical Hamiltonian at time t.
        """
        return self.envelope(t) * self.carrier(t)

class simulation:

    def __init__(self, system, evolver, nt, dt):
        """
        Create an instance of a simulation that calculates the time evolution of a single atom interacting
        with lasers at fixed frequencies.
        :param system: An instance of the class atom.
        :param evolver: The method hamiltonian from the class hamiltonian_construct.
        :param nt: Number of time steps in the simulation.
        :param dt: Time step.
        """
        self.system = system
        self.evolver = evolver
        self.nt = nt
        self.dt = dt

    def reset_system(self):
        """
        Return the system to its original state.
        :return: None.
        """
        self.system.current_state = self.system.initial_state
        return None

    def final_state(self):
        """
        Calculate the state of the system after nt timesteps of size dt.
        :return: The final state of the system (matrix).
        """
        times = sp.linspace(0, self.nt * self.dt, self.nt)
        for i in times:
            self.system.evolve_step(self.evolver(i * self.dt), self.dt)

        return self.system.current_state

    def time_evolution(self):
        """
        Calculate the state of the system at each of the nt time steps of size dt.
        :return: The state at ever timestep of the simulation.
        """

        row = sp.shape(self.system.inital_state)[0]
        column = sp.shape(self.system.inital_state)[0]
        times = sp.linspace(0, self.nt * self.dt, self.nt)
        time_dep_state = sp.zeros((self.nt, row, column), dtype=complex)

        for i in times:
            time_dep_state[int(i / self.dt), :, :] = self.system.evolve_step(self.evolver(i * self.dt), self.dt).copy()

        return time_dep_state