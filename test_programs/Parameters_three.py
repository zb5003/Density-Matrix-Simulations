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
n_states = 3
initial_state = sp.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)

gamma = 2 * sp.pi * 2e6
gamma_slow = 2 * sp.pi * 1e5
decay_matrix = sp.asarray([[0, 0, 0], [0, gamma_slow, 0], [0, 0, gamma]])

decay_to = sp.asarray([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])

ionic_density = 9.35e24  # /m^3import
n_151 = 800
n_153 = 800
isotope_names = ['151', '153']
number_of_atoms = [n_151, n_153]
n_total = sum(number_of_atoms)
ib_linewidth = 2 * sp.pi * n_total * 2500000  # In Hz

# Beam parameters
power_p = 0.0001
waist_p = 2000e-6
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(2 * intensity_p / (n_refraction * epsilon0 * c))
Rabi_p = a0 * e_charge * field_amplitude_p / hbar

power_c = 0.0005
waist_c = 2000e-6
intensity_c = power_c / (2 * sp.pi * waist_c**2)
field_amplitude_c = sp.sqrt(2 * intensity_c / (n_refraction * epsilon0 * c))
Rabi_c = a0 * e_charge * field_amplitude_c / hbar

# Interaction parameters
detuning_p = 0
frequencies_p = sp.asarray([[0, 0, detuning_p], [0, 0, 0], [-detuning_p, 0, 0]])
detunings_p = sp.linspace(-10 * gamma, 10 * gamma, 100)
dipole_operator_p = a0 * e_charge * sp.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

detuning_c = 0
frequencies_c = sp.asarray([[0, 0, 0], [0, 0, detuning_c], [0, -detuning_c, 0]])
dipole_operator_c = a0 * e_charge * sp.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0]])


# Simulation parameters
dt = 1 / (10 * max([Rabi_c, Rabi_p, detuning_p, detuning_c]))
nt = int((20 * (2 * sp.pi) / gamma) / dt)
print(dt * 1e9, nt)
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)
