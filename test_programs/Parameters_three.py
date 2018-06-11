from density_matrix_classes.physicalconstants import *
import scipy as sp

def pulse_sequence(t, tau, tau_0):
    return 1 / (1 + sp.exp(-(t - tau_0) / tau)) - 1 / (1 + sp.exp(-(t - tau_0 - 2e-6) / tau)) + 50 * (1 / (1 + sp.exp(-(t - tau_0 - 5e-6) / tau)) - 1 / (1 + sp.exp(-(t - tau_0 - 5e-6 - 1e-8) / tau)))

def frequency_matrix_generator(detuning, lower, upper):
    """
    Create the frequency matrix for an atom with particular level splittings and a beam with a particular detuning.
    The (i, j)th element of the matrix represents the detuning of the i to j transition from the beam.
    The lower left of the matrix has negative frequencies for red detunings and positive frequencies for blue detunings.
    The upper right is the opposite.
    
    By default the target transition is the lowest ground state level to the lowest excited state level.
    The matrix elements corresponding to these transition will be zero for detuning=0.
    :param detuning: Float. The detuning of the beam from the target transition. Negative for red transitions, positive for blue.
    :param lower: List/Array.  Frequency splittings of the ground state. The first element is the splitting between the lowest two levels,
                  the second element is the splitting between the second and third lowest states...
    :param upper: List/Array. Same as lower except for the excited states.
    :return: Array. This array contains the frequencies that the Hamiltonian matrix elements will oscillate at.
    """
    detuning = -detuning
    nrows = len(lower) + 1
    ncols = len(upper) + 1
    intermediate = sp.full((nrows, ncols), detuning, dtype=float)
    for index_i in range(nrows):
        for index_j in range(ncols):
            intermediate[index_i, index_j] = intermediate[index_i, index_j] - sum(lower[0:index_i]) + sum(upper[0:index_j])
    temp1 = sp.hstack((sp.zeros((nrows, nrows)), intermediate))
    temp2 = sp.hstack((-intermediate.T, sp.zeros((ncols, ncols))))
    final = sp.vstack((temp1, temp2))
    return final

filename = "Three_Level"

n_refraction = 1.8
n_states = 3
initial_state = sp.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)

gamma = 2 * sp.pi * 4800
gamma_slow = 2 * sp.pi * 500
decay_matrix = sp.asarray([[0, 0, 0], [0, gamma_slow, 0], [0, 0, gamma]])

decay_to = sp.asarray([[0, sp.sqrt(gamma_slow), sp.sqrt(gamma / 2)],
                       [0, 0, sp.sqrt(gamma / 2)],
                       [0, 0, 0]])

ionic_density = 9.35e24  # /m^3import
n_151 = 800
n_153 = 800
isotope_names = ['151', '153']
number_of_atoms = [n_151, n_153]
n_total = sum(number_of_atoms)
ib_linewidth = 2 * sp.pi * n_total * 2500000  # In Hz

# Beam parameters
power_p = 10e-6
waist_p = 50e-6
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(2 * intensity_p / (n_refraction * epsilon0 * c))
field_amplitude_p_B = sp.sqrt(4 * mu0 * n_refraction * power_p / (c * sp.pi * waist_p**2))
Rabi_p = 0.063 * muB * field_amplitude_p_B / hbar

power_c = 0.05
waist_c = 100e-6
intensity_c = power_c / (2 * sp.pi * waist_c**2)
field_amplitude_c = sp.sqrt(2 * intensity_c / (n_refraction * epsilon0 * c))
field_amplitude_c_B = sp.sqrt(4 * mu0 * n_refraction * power_c / (c * sp.pi * waist_c**2))
Rabi_c = 0.063 * muB * field_amplitude_c_B / hbar
print(round(Rabi_p / 1e3, 3), round(Rabi_c / 1e3, 3))
# Interaction parameters
lower_spacing = [100e6]
upper_spacing = []

detuning_p = 0
frequencies_p = frequency_matrix_generator(detuning_p, lower_spacing, upper_spacing)  # sp.asarray([[0, 0, detuning_p], [0, 0, 0], [-detuning_p, 0, 0]])
detunings_p = sp.linspace(-5 * gamma, 5 * gamma, 300)
dipole_operator_p = a0 * e_charge * sp.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]])


detuning_c = -lower_spacing[0]
frequencies_c = frequency_matrix_generator(detuning_c, lower_spacing, upper_spacing)
dipole_operator_c = a0 * e_charge * sp.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
print(frequencies_p / 1e6, frequencies_c / 1e6)

# Simulation parameters
dt = 0.1e-9  # 1 / (800 * max([Rabi_c, Rabi_p, detuning_p, detuning_c]))
nt = 30000  # int((1 * (2 * sp.pi) / gamma) / dt)
print("Time step =", dt * 1e9, "ns; Number of time steps =", nt)
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)
