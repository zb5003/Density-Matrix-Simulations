from density_matrix_classes.physicalconstants import *
import scipy as sp

def frequency_matrix_generator(detuning, lower, upper):
    """

    :param detuning:
    :param lower:
    :param upper:
    :return:
    """
    nrows = len(lower) + 1
    ncols = len(upper) + 1
    intermediate = sp.full((nrows, ncols), detuning, dtype=float)
    for index_i in range(nrows):
        for index_j in range(ncols):
            intermediate[index_i, index_j] = intermediate[index_i, index_j] - sum(lower[0:index_i]) + sum(upper[0:index_j])
    temp1 = sp.hstack((sp.zeros((nrows, nrows + 1)), intermediate))
    temp2 = sp.vstack((temp1, sp.zeros((1, nrows + ncols + 1))))
    temp3 = sp.hstack((-intermediate.T, sp.zeros((ncols, ncols + 1))))
    final = sp.vstack((temp2, temp3))
    return final

filename = "Seven_Level"

n_refraction = 1.8
n_states = 7
initial_state = sp.asarray([[1 / 3, 0, 0, 0, 0, 0, 0],
                            [0, 1 / 3, 0, 0, 0, 0, 0],
                            [0, 0, 1 / 3, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=complex)
gamma = 2 * sp.pi * 4800
gamma_slow = 2 * sp.pi * 500
decay_matrix = sp.asarray([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, gamma_slow, 0, 0, 0],
                           [0, 0, 0, 0, gamma, 0, 0],
                           [0, 0, 0, 0, 0, gamma, 0],
                           [0, 0, 0, 0, 0, 0, gamma]])
decay_to = sp.asarray([[0, 0, 0, sp.sqrt(gamma_slow / 3), 0, 0, 0],
                       [0, 0, 0, sp.sqrt(gamma_slow / 3), 0, 0, 0],
                       [0, 0, 0, sp.sqrt(gamma_slow / 3), 0, 0, 0],
                       [0, 0, 0, 0, sp.sqrt(gamma), sp.sqrt(gamma), sp.sqrt(gamma)],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0]])

ionic_density = 9.35e24  # /m^3import
n_151 = 800
n_153 = 800
isotope_names = ['151', '153']
number_of_atoms = [n_151, n_153]
n_total = sum(number_of_atoms)
ib_linewidth = 2 * sp.pi * n_total * 2500000  # In Hz

# Beam parameters
power = 277e-3
waist = 112e-6 / 2
intensity = power / (2 * sp.pi * waist**2)
field_amplitude = sp.sqrt(4 * mu0 * n_refraction * power / (c * sp.pi * waist**2))

# Interaction parameters
detuning = 0
lower_spacing_151 = 2 * sp.pi * sp.asarray([57.3 * 10**6, 29.5 * 10**6], dtype=sp.float64)
upper_spacing_151 = 2 * sp.pi * sp.asarray([71 * 10**6, 43 * 10**6], dtype=sp.float64)
frequencies_151 = frequency_matrix_generator(detuning, lower_spacing_151, upper_spacing_151)
lower_spacing_153 = sp.asarray([148 * 10**6, 76.4 * 10**6], dtype=sp.float64)
upper_spacing_153 = sp.asarray([183 * 10**6, 114 * 10**6], dtype=sp.float64)
frequencies_153 = frequency_matrix_generator(detuning, lower_spacing_153, upper_spacing_153)
frequencies = [frequencies_151, frequencies_153]
detunings = sp.linspace(-10 * gamma, 10 * gamma, 100)
dipole_operator = 0.063 * muB * sp.asarray([[0, 0, 0, 0, sp.sqrt(0.03), sp.sqrt(0.22), sp.sqrt(0.75)],
                                            [0, 0, 0, 0, sp.sqrt(0.12), sp.sqrt(0.68), sp.sqrt(0.2)],
                                            [0, 0, 0, 0, sp.sqrt(0.85), sp.sqrt(0.1), sp.sqrt(0.05)],
                                            [0, 0, 0, 0, 0, 0, 0],
                                            [sp.sqrt(0.03), sp.sqrt(0.12), sp.sqrt(0.85), 0, 0, 0, 0],
                                            [sp.sqrt(0.22), sp.sqrt(0.68), sp.sqrt(0.10), 0, 0, 0, 0],
                                            [sp.sqrt(0.75), sp.sqrt(0.2), sp.sqrt(0.05), 0, 0, 0, 0]])
Rabi_f = a0 * e_charge * field_amplitude / hbar

# Simulation parameters
dt = 1e-10
nt = 15000
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)