from density_matrix_classes.physicalconstants import *
import scipy as sp

n_refraction = 1.8
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

number_of_atoms = 1
ib_linewidth = 0

# Beam parameters
power = 277e-3
waist = 112e-6 / 2
intensity = power / (2 * sp.pi * waist**2)
field_amplitude = sp.sqrt(4 * mu0 * n_refraction * power / (c * sp.pi * waist**2))

# Interaction parameters
detuning = 0
lower_spacing = sp.asarray([148 * 10**6, 76.4 * 10**6], dtype=sp.float64)
upper_spacing = sp.asarray([183 * 10**6, 114 * 10**6], dtype=sp.float64)
frequencies_intermediate = sp.asarray([[detuning, detuning + upper_spacing[0], detuning + sum(upper_spacing)],
                            [detuning - lower_spacing[0], detuning + upper_spacing[0] - lower_spacing[0], detuning + sum(upper_spacing) - lower_spacing[0]],
                            [detuning - sum(lower_spacing), detuning + upper_spacing[0] - sum(lower_spacing), detuning + sum(upper_spacing) - sum(lower_spacing)]])

frequencies = sp.vstack((sp.hstack((sp.zeros((4, 4)), sp.vstack((frequencies_intermediate, sp.zeros(3))))), sp.hstack((-frequencies_intermediate, sp.zeros((3, 4))))))
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
dt = 1e-9
nt = 3000
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)