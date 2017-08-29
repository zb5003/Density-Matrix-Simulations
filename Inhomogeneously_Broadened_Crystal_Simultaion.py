"""
All units are in SI.
"""
import scipy as sp
import time
import matplotlib.pyplot as plt
import shutil
from Atomic_Simulation_Classes import *
from Inhomogeneous_Broadening_Classes import *
from Data_Saving_Functions import *
from physicalconstants import *

# Atomic parameters _____________________________
n = 1.8
lower_spacing_153 = 2 * sp.pi * sp.asarray([148.1 * 10**6, 76.4 * 10**6], dtype=sp.float64)
upper_spacing_153 = 2 * sp.pi * sp.asarray([183 * 10**6, 114 * 10**6], dtype=sp.float64)
lower_spacing_151 = 2 * sp.pi * sp.asarray([57.3 * 10**6, 29.5 * 10**6], dtype=sp.float64)
upper_spacing_151 = 2 * sp.pi * sp.asarray([71 * 10**6, 43 * 10**6], dtype=sp.float64)
initial_state = sp.asarray([[1 / 3, 0, 0, 0, 0, 0, 0],
                            [0, 1 / 3, 0, 0, 0, 0, 0],
                            [0, 0, 1 / 3, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], dtype=complex)

gamma = 1 / 33e-6
gamma_slow = 1 / 1.61e-3
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
# print(decay_to)
# decay_to = sp.zeros((7, 7))

number_of_atoms = 400 #40000
ib_linewidth = 2 * sp.pi * 1e9 #number_of_atoms * 25000

# Beam parameters _______________________________
power_p = 277e-3
waist_p = 112e-6 / 2
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(4 * mu0 * n * power_p / (c * sp.pi * waist_p**2))
Rabi = 0.063 * muB * field_amplitude_p / hbar

# power_c = 0.05
# waist_c = 2000e-6
# intensity_c = power_c / (2 * sp.pi * waist_c**2)
# field_amplitude_c = sp.sqrt(2 * intensity_c / (n * epsilon0 * c))

# Interaction parameters ________________________
detuning = -35 * 1e6
frequencies_intermediate_153 = sp.asarray([[detuning, detuning + upper_spacing_153[0], detuning + sum(upper_spacing_153)],
                            [detuning - lower_spacing_153[0], detuning + upper_spacing_153[0] - lower_spacing_153[0], detuning + sum(upper_spacing_153) - lower_spacing_153[0]],
                            [detuning - sum(lower_spacing_153), detuning + upper_spacing_153[0] - sum(lower_spacing_153), detuning + sum(upper_spacing_153) - sum(lower_spacing_153)]])

frequencies_p_153 = sp.vstack((sp.hstack((sp.zeros((4, 4)), sp.vstack((frequencies_intermediate_153, sp.zeros(3))))), sp.hstack((-frequencies_intermediate_153.T, sp.zeros((3, 4))))))


frequencies_intermediate_151 = sp.asarray([[detuning, detuning + upper_spacing_151[0], detuning + sum(upper_spacing_151)],
                            [detuning - lower_spacing_151[0], detuning + upper_spacing_151[0] - lower_spacing_151[0], detuning + sum(upper_spacing_151) - lower_spacing_151[0]],
                            [detuning - sum(lower_spacing_151), detuning + upper_spacing_151[0] - sum(lower_spacing_151), detuning + sum(upper_spacing_151) - sum(lower_spacing_151)]])

frequencies_p_151 = sp.vstack((sp.hstack((sp.zeros((4, 4)), sp.vstack((frequencies_intermediate_151, sp.zeros(3))))), sp.hstack((-frequencies_intermediate_151.T, sp.zeros((3, 4))))))

detunings_p = sp.linspace(-10 * gamma, 10 * gamma, 100)
# dipole_operator_p = 0.063 * muB * sp.asarray([[0, 0, 0, 0, sp.sqrt(0.03), sp.sqrt(0.22), sp.sqrt(0.75)],
#                                               [0, 0, 0, 0, sp.sqrt(0.12), sp.sqrt(0.68), sp.sqrt(0.2)],
#                                               [0, 0, 0, 0, sp.sqrt(0.85), sp.sqrt(0.1), sp.sqrt(0.05)],
#                                               [0, 0, 0, 0, 0, 0, 0],
#                                               [sp.sqrt(0.03), sp.sqrt(0.12), sp.sqrt(0.85), 0, 0, 0, 0],
#                                               [sp.sqrt(0.22), sp.sqrt(0.68), sp.sqrt(0.10), 0, 0, 0, 0],
#                                               [sp.sqrt(0.75), sp.sqrt(0.2), sp.sqrt(0.05), 0, 0, 0, 0]])
dipole_operator_p = 0.063 * muB * sp.asarray([[0, 0, 0, 0, sp.sqrt(0.85), sp.sqrt(0.1), sp.sqrt(0.05)],
                                              [0, 0, 0, 0, sp.sqrt(0.12), sp.sqrt(0.68), sp.sqrt(0.2)],
                                              [0, 0, 0, 0, sp.sqrt(0.03), sp.sqrt(0.22), sp.sqrt(0.75)],
                                              [0, 0, 0, 0, 0, 0, 0],
                                              [sp.sqrt(0.85), sp.sqrt(0.12), sp.sqrt(0.03), 0, 0, 0, 0],
                                              [sp.sqrt(0.10), sp.sqrt(0.68), sp.sqrt(0.22), 0, 0, 0, 0],
                                              [sp.sqrt(0.05), sp.sqrt(0.2), sp.sqrt(0.75), 0, 0, 0, 0]])

# frequencies_c = sp.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# dipole_operator_c = a0 * e_charge * sp.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

# Simulation parameters _________________________
dt = 1e-9
nt = 2100
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)

# Objects _____________________________________________________________________
the_atom_153 = atom(initial_state, decay_matrix, decay_to)
the_hamiltonian_p_153 = hamiltonian_construct(dipole_operator_p, field_amplitude_p, frequencies_p_153)
# the_hamiltonian_c = hamiltonian_construct(dipole_operator_c, field_amplitude_c, frequencies_c)
the_simulation_153 = single_atom_simulation(the_atom_153, [the_hamiltonian_p_153], nt, dt)
inhomogeneously_broadened_simulation_153 = inhomogeneous_broadening(the_simulation_153, ib_linewidth, number_of_atoms)

the_atom_151 = atom(initial_state, decay_matrix, decay_to)
the_hamiltonian_p_151 = hamiltonian_construct(dipole_operator_p, field_amplitude_p, frequencies_p_151)
# the_hamiltonian_c = hamiltonian_construct(dipole_operator_c, field_amplitude_c, frequencies_c)
the_simulation_151 = single_atom_simulation(the_atom_151, [the_hamiltonian_p_151], nt, dt)
inhomogeneously_broadened_simulation_151 = inhomogeneous_broadening(the_simulation_151, ib_linewidth, number_of_atoms)

# Run the simulation __________________________________________________________
t11 = time.time()
the_flop_153 = inhomogeneously_broadened_simulation_153.broadened_time_evolution()
the_flop_151 = inhomogeneously_broadened_simulation_151.broadened_time_evolution()
the_flop = (the_flop_151 + the_flop_153) / 2
# the_susceptibility = the_simulation.susceptibility(detunings_p)
print("Time elapsed = " + str(round(time.time() - t11, 4)) + " seconds")
print("Rabi frequency =", Rabi, "MHz; Rabi period =", round(1 / Rabi * 1e6, 3), "microseconds")

# Save dat shit _______________________________________________________________
loc = file_manager("Seven_Level_Inhomogeneously_Broadened")
populations_plot(the_times * 1e6, the_flop, loc)
crystal_pop_compare(the_times * 1e6, the_flop, loc)
coherence_plot(the_times * 1e6, the_flop, loc)
ground_v_excited_7(the_times * 1e6, the_flop, loc)
total_coherence_7(the_times * 1e6, the_flop, loc)
shutil.copy("Crystal_Simulation.py", loc)
sp.save(loc + "/data.txt", the_flop)  # This could be a pretty big file
sp.save(loc + "/times.txt", the_times)
