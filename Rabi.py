import densitymatrix.nlevel as nl
import physicalconstants as pc
import scipy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.random as rnd
import multiprocessing as mp
import os
import time

class Rabi_Flopping:

    version = 1

    def __init__(self, field, dipoles, lower_detunings, upper_detunings, targets, gamma, decay_to, open_system="no"):
        """
        Class for computing and displaying the time dependence of a Rabi Flopping simulation.
        Dependent on nlevel.AL_interaction.
        :param field: Field amplitude of Rabi beam.
        :param dipoles: Matrix of dipole moments between states.
        :param lower_detunings: Spacings of the lower levels (contains one less than the total number of lower states).
        :param upper_detunings: Spacings of the upper states (contains one less than the total number of upper states).
        :param targets: The two states to be addressed if there were no inhomogeneous broadening (one upper, one lower).
        :param gamma: Decay matrix from each state.
        :param decay_to: Matrix whose elements are decay rates into a state from another state. For example,
                         Decay_to[i, j] is the decay rate from state j to state i.
        :param open_system:
        """
        self.field = field
        self.dipoles = dipoles
        self.lower_detunings = lower_detunings
        self.upper_detunings = upper_detunings
        self.targets = targets
        self.gamma = gamma
        self.decay_to = decay_to
        self.open_system = open_system

        self.n_lower = len(lower_detunings) + 1  # number of lower states
        self.n_upper = len(upper_detunings) + 1  # number of upper states
        self.n = len(lower_detunings) + len(upper_detunings) + 2  # total number of states

        self.n_interaction = self.generate_n_interaction(dipoles)  # for use with off_diagonal_average===
        self.check()

        self.n_cores = mp.cpu_count() - 1  # number of cores available for multiprocessing

    def generate_n_interaction(self, mat):
        """
        Find the total number of interacting states (the number of nonzero off-diagonal elements divided by two).
        :param mat: Interaction matrix.
        :return: Number of interacting states.
        """
        summ = len(sp.nonzero(mat)[0])

        return summ / 2

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        """
        Check the validity the values entered for the attribute "targets." There should be two target states, each of
        which must be an existing state.  The smaller value in target is the lower state and the larger value is the
        excited state.  This is a setter isntead of a method like check because the target states might have to be
        reorganized.
        :param value: The intended target states.
        :return: None.
        """
        if len(value) != 2 or len(sp.shape(value)) != 1:
            raise Exception("There must be two target states.")
        elif value[0] <= 0 or value[1] <= 0 or value[0] > len(self.dipoles) or value[1] > len(self.dipoles):
            raise Exception("Target states must be greater than zero and less than or equal to the number of states.")
        elif value[0] >= value[1]:
            temp = value[0]
            value[0] = value[1]
            value[1] = temp
            self._targets = value
        else:
            self._targets = value
        return None

    def check(self):
        """
        Checks to make sure attribute of an instance are of the proper form before continuing with any calculations.
        The dipole matrix should be an NxN matrix where N > 0 and is the total number of states in the system.
        N should be equal to the sum of the detunings plus two.
        The gamma matrix should also be an NxN matrix.
        The decay_to matrix should also be NxN.
        :return: None
        """
        if len(sp.shape(self.dipoles)) != 2 or sp.shape(self.dipoles)[0] < 2:
            raise Exception("The argument dipoles must be a square matrix that is at least 2x2.")

        if len(self.lower_detunings) + len(self.upper_detunings) != sp.shape(self.dipoles)[0] - 2:
            raise Exception("The number of state detunings does not match the number of states.")

        if len(sp.shape(self.gamma)) != 2 or sp.shape(self.gamma)[0] < 2:
            raise Exception("The argument gamma must be a square matrix that is at least 2x2.")

        if len(sp.shape(self.decay_to)) != 2 or sp.shape(self.decay_to)[0] < 2:
            raise Exception("The argument decay_to must be a square matrix that is at least 2x2.")

        return None

    def off_diagonal_average(self, data):
        """
        Average the off diagonal elements of the density matrix at each timestep.  Can be used to calculate the
        magnetization or polarization.
        :param data: Simulation data.
        :return: Average of the off diagonal matrix elements for each timestep.
        """

        average = sp.zeros((sp.shape(data)[0]))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                average = average + data[:, i, j]
        average = average / self.n_interaction

        return average

    def time_evolve(self, rho, detuning, dur, dt, shift=0):
        """
        Carry out the time evolution of the system.
        :param rho: Initial density matrix.
        :param detuning: Detuning of the driving beam (in Hz).
        :param dur: Duration of the simulation (in seconds).
        :param dt: Time step (in seconds).
        :param shift: Shift in the energy levels (in Hz).
        :return: Time evolved density matrix at each time step.
        """
        nt = int(dur / dt)
        V_pert = nl.AL_interaction(self.field, self.dipoles, self.lower_detunings, self.upper_detunings, self.targets,
                                   detuning, shift)
        evolve = nl.NLevel_Evolve(rho, self.gamma, self.decay_to, self.open_system)
        rho_t = sp.zeros((nt, sp.shape(rho)[0], sp.shape(rho)[1]), dtype=complex)

        for i in range(nt):
            tem = evolve.RK_rho(V_pert.interaction(i * dt), rho, dt)
            rho = tem.copy()
            transform = V_pert.oscillations((-i * dt))
            rho_t[i] = sp.multiply(rho, transform) + sp.diag(sp.diag(rho))  # Remove underlying osc. due to Ham.

        return rho_t

    def final_state(self, rho, detuning, dur, dt, shift=0):
        """
        Calculate the final density matrix of the system after evolving for a time dur.
        :param rho: Initial density matrix.
        :param detuning: Detuning of the driving beam (in Hz).
        :param dur: Duration of the simulation (in seconds).
        :param dt: Time step (in seconds).
        :param shift: Shift in the energy levels (in Hz).
        :return: The final density matrix.
        """
        nt = int(dur / dt)
        V_pert = nl.AL_interaction(self.field, self.dipoles, self.lower_detunings, self.upper_detunings, self.targets,
                                   detuning, shift)
        evolve = nl.NLevel_Evolve(rho, self.gamma, self.decay_to, self.open_system)
        for i in range(nt):
            tem = evolve.RK_rho(V_pert.interaction(i * dt), rho, dt)
            rho = tem.copy()
        transform = V_pert.oscillations(-(nt - 1) * dt)
        rho = sp.multiply(rho, transform) + sp.diag(sp.diag(rho))  # Remove underlying osc. due to Ham.
        return rho

    def susceptibility(self, rho, low, high, N, dt, dur, shift=0, points=100):
        """
        Calculate the susceptibility for a range of detunings.
        :param rho: Initial density matrix
        :param low: Lower limmit for detunings
        :param high: Upper limit for detunings
        :param N: Atomic density
        :param dt: Time step
        :param dur: Duration of simulation for each detuning. Must be long enough to reach steady state. Default is 1
        :param points: Number of detunings
        :return: 1-D array containing the susceptibility at each frequency
        """
        coef = N * self.dipoles[self.targets[0] - 1, self.targets[1] - 1] / (pc.epsilon0 * self.field)
        response = sp.zeros(points, dtype=complex)
        detunings = sp.linspace(low, high, points)

        for i in range(points):
            evolved = self.final_state(rho, detunings[i], dur, dt, shift)
            response[i] = coef * evolved[self.targets[0] - 1, self.targets[1] - 1]
            print(i)

        return detunings, response

    def susceptibility_parallel(self, rho, low, high, N, dt, dur, shift=0, points=100):
        """
        Calculate the susceptibility for a range of detunings.  This method makes use of the multiprocessing module.
        A pool is created and the method pool.apply_async is used to keep the detunings in order while still giving a
        substantial speedup.
        :param rho: Initial density matrix
        :param low: Lower limmit for detunings (Hz)
        :param high: Upper limit for detunings (Hz)
        :param N: Atomic density
        :param dt: Time step
        :param dur: Duration of simulation for each detuning. Must be long enough to reach steady state. Default is 1
        :param points: Number of detunings
        :return: 1-D array containing the susceptibility at each frequency
        """
        coef = N * self.dipoles[self.targets[0] - 1, self.targets[1] - 1] / (pc.epsilon0 * self.field)
        detunings = sp.linspace(low, high, points)

        pool = mp.Pool(processes=self.n_cores)
        out = [pool.apply_async(self.final_state, args=(rho, detunings[i], dur, dt, shift)) for i in range(points)]
        results = sp.asarray([coef * p.get()[self.targets[0] - 1, self.targets[1] - 1] for p in out])

        return detunings, results

class inhomogeneous_broadening(Rabi_Flopping):

    version = 1
    trunc = 5

    def __init__(self, width, n_samples,  field, dipoles, lower_detunings, upper_detunings, targets, gamma, decay_to,
                 open_system="no"):
        Rabi_Flopping.__init__(self, field, dipoles, lower_detunings, upper_detunings, targets, gamma, decay_to,
                               open_system)
        self.width = width
        self.n_samples = n_samples
        self.samples = sp.linspace(-width, width, n_samples)  # self.line_profile()
        self.directory = "RF_LVL_" + str(self.targets[0]) + str(self.targets[1]) + "_IB_ATOM" + str(self.n_samples)+\
                         "_STD"+str(int(self.width / 10**6))+"_SAM"+str(self.n_samples)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.n_cores = mp.cpu_count() - 1  # Number of cores available for multiprocessing


    def line_profile(self):
        """
        Generate all the energy level shifts for each of the self.n_samples ions in the simulation.  They are spread out
        over a Gaussian inhomogeneous broadened line of width self.width.
        :return: A 1-D array containing randomly sampled energy shifts from the inhomogeneous line.
        """
        rnd.seed(0)

        return rnd.normal(0, self.width, self.n_samples)

    def inhomogeneously_broadened_time_evolution(self, rho, dur, dt):
        """
        Calculate the time evolution of the density matrix across an inhomogeneously broadened transition. The raw data
        as well as the averaged off-diagonal elements are saved to .npy files.
        :param rho: Inital density matrix.
        :param dur: Duration of the simulation (in seconds).
        :param dt: Time step (in seconds).
        :return: The density matrix, averaged over the inhomogeneous line, for each time step.
        """
        nt = int(dur / dt)
        shifts = self.samples
        response = sp.zeros((nt, sp.shape(rho)[0], sp.shape(rho)[1]), dtype=complex)
        for i in range(self.n_samples):
            t_start = time.time()
            temp1 = self.time_evolve(rho, 0, dur, dt, shifts[i])
            response = response + temp1
            print("step " + str(i) + " took " + str(round(time.time() - t_start, 4)) + " seconds")

        response = response / self.n_samples
        # sp.save(self.directory + "/DIAG", np.round(response, inhomogeneous_broadening.trunc))

        return response  # For plotting purposes

    def inhomogeneously_broadened_time_evolution_parallel(self, rho, dur, dt):
        """
        Calculate the time evolution of the density matrix across an inhomogeneously broadened transition. The raw data
        as well as the averaged off-diagonal elements are saved to .npy files.  This method makes use of the
        multiprocessing module. A pool is created and the method pool.apply_async is used to keep the detunings in order while still giving a
        substantial speedup.
        :param rho: Inital density matrix.
        :param dur: Duration of the simulation (in seconds).
        :param dt: Time step (in seconds).
        :return: The density matrix, averaged over the inhomogeneous line, for each time step.
        """
        shifts = self.samples
        pool = mp.Pool(processes=self.n_cores)
        out = [pool.apply_async(self.time_evolve, args=(rho, 0, dur, dt, shifts[i])) for i in range(self.n_samples)]
        extract = sp.asarray([p.get() for p in out])
        response = sum(extract)

        response = response / self.n_samples
        # sp.save(self.directory + "/DIAG", np.round(response, inhomogeneous_broadening.trunc))

        return response  # For plotting purposes

    def inhomogeneously_broadened_population_distribution(self, rho, dt, dur, shift=0):
        """

        :param rho:
        :param low:
        :param high:
        :param N:
        :param dt:
        :param dur:
        :param shift:
        :return:
        """
        shifts = self.samples
        split = 1  # number of samples in each bin
        bins = math.ceil(self.n_samples / split)  # number of bins
        det = sp.linspace(-self.width, self.width, bins)
        response = sp.zeros((bins, sp.shape(rho)[0], sp.shape(rho)[1]), dtype=complex)
        pop_dist = sp.zeros((bins, sp.shape(rho)[0], sp.shape(rho)[1]), dtype=complex)
        for i in range(self.n_samples):
            response[i] = self.final_state(rho, 0, dur, dt, shifts[i])
            pop_dist[int(i / split)] = pop_dist[int(i / split)] + response[i]
            print(i)

        return det, response, pop_dist  # For plotting purposes

    def inhomogeneously_broadened_population_distribution_parallel(self, rho, dt, dur, shift=0):
        """

        :param rho:
        :param low:
        :param high:
        :param N:
        :param dt:
        :param dur:
        :param shift:
        :return:
        """
        shifts = self.samples
        pool = mp.Pool(processes=self.n_cores)
        out = [pool.apply_async(self.final_state, args=(rho, 0, dur, dt, shifts[i])) for i in range(self.n_samples)]
        extract = sp.asarray([p.get() for p in out], dtype=complex)

        split = 1  # number of samples in each bin
        bins = math.ceil(self.n_samples / split)  # number of bins
        det = sp.linspace(-self.width, self.width, bins)
        pop_dist = sp.zeros((bins, sp.shape(rho)[0], sp.shape(rho)[1]), dtype=complex)
        for i in range(self.n_samples):
            pop_dist[int(i / split)] = pop_dist[int(i / split)] + extract[i]

        # add a normalization for the bins
        # center the bin frequency
        # maybe save data to a file

        return det, pop_dist  # For plotting purposes


