"""
All units are in SI.
"""
import scipy as sp
import time
import matplotlib.pyplot as plt
import shutil
from density_matrix_classes.Atomic_Simulation_Classes import *
from density_matrix_classes.Data_Saving_Functions import *
from density_matrix_classes.physicalconstants import *

# Atomic parameters _____________________________
n = 1.8
lower_spacing = sp.asarray([148 * 10**6, 76.4 * 10**6], dtype=sp.float64)
upper_spacing = sp.asarray([183 * 10**6, 114 * 10**6], dtype=sp.float64)
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

# Beam parameters _______________________________
power_p = 277e-3
waist_p = 112e-6 / 2
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(4 * mu0 * n * power_p / (c * sp.pi * waist_p**2))
Rabi = 0.063 * muB * field_amplitude_p / hbar
print(Rabi, 1 / Rabi)

# power_c = 0.05
# waist_c = 2000e-6
# intensity_c = power_c / (2 * sp.pi * waist_c**2)
# field_amplitude_c = sp.sqrt(2 * intensity_c / (n * epsilon0 * c))

# Interaction parameters ________________________
detuning = 0
frequencies_intermediate = sp.asarray([[detuning, detuning + upper_spacing[0], detuning + sum(upper_spacing)],
                            [detuning - lower_spacing[0], detuning + upper_spacing[0] - lower_spacing[0], detuning + sum(upper_spacing) - lower_spacing[0]],
                            [detuning - sum(lower_spacing), detuning + upper_spacing[0] - sum(lower_spacing), detuning + sum(upper_spacing) - sum(lower_spacing)]])

frequencies_p = sp.vstack((sp.hstack((sp.zeros((4, 4)), sp.vstack((frequencies_intermediate, sp.zeros(3))))), sp.hstack((-frequencies_intermediate, sp.zeros((3, 4))))))

detunings_p = sp.linspace(-10 * gamma, 10 * gamma, 100)
dipole_operator_p = 0.063 * muB * sp.asarray([[0, 0, 0, 0, sp.sqrt(0.03), sp.sqrt(0.22), sp.sqrt(0.75)],
                                              [0, 0, 0, 0, sp.sqrt(0.12), sp.sqrt(0.68), sp.sqrt(0.2)],
                                              [0, 0, 0, 0, sp.sqrt(0.85), sp.sqrt(0.1), sp.sqrt(0.05)],
                                              [0, 0, 0, 0, 0, 0, 0],
                                              [sp.sqrt(0.03), sp.sqrt(0.12), sp.sqrt(0.85), 0, 0, 0, 0],
                                              [sp.sqrt(0.22), sp.sqrt(0.68), sp.sqrt(0.10), 0, 0, 0, 0],
                                              [sp.sqrt(0.75), sp.sqrt(0.2), sp.sqrt(0.05), 0, 0, 0, 0]])

# frequencies_c = sp.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# dipole_operator_c = a0 * e_charge * sp.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

# Simulation parameters _________________________
dt = 1e-9
nt = 3000
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)

# Objects _____________________________________________________________________
the_atom = atom(initial_state, decay_matrix, decay_to)
the_hamiltonian_p = hamiltonian_construct(dipole_operator_p, field_amplitude_p, frequencies_p)
# the_hamiltonian_c = hamiltonian_construct(dipole_operator_c, field_amplitude_c, frequencies_c)
the_simulation = single_atom_simulation(the_atom, [the_hamiltonian_p], nt, dt)

# Run the simulation __________________________________________________________
t1 = time.time()
the_flop = the_simulation.time_evolution_serial()
# the_susceptibility = the_simulation.susceptibility(detunings_p)
print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")

# Save dat shit _______________________________________________________________
loc = file_manager("Seven_Level")
populations_plot(the_times * 1e6, the_flop, loc)
crystal_pop_compare(the_times * 1e6, the_flop, loc)
coherence_plot(the_times * 1e6, the_flop, loc)
shutil.copy("Crystal_Simulation.py", loc)
sp.save(loc + "/data.txt", the_flop)
sp.save(loc + "/times.txt", the_times)

