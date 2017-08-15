"""
All units are in SI.
"""
import scipy as sp
import time
import matplotlib.pyplot as plt
from Atomic_Simulation_Classes import *
from physicalconstants import *

# Atomic parameters _____________________________
n = 1.8
lower_spacing = sp.asarray([148 * 10**6, 76.4 * 10**6], dtype=sp.float64)
upper_spacing = sp.asarray([183 * 10**6, 114 * 10**6], dtype=sp.float64)
initial_state = sp.asarray([[1 / 3, 0, 0, 0, 0, 0],
                            [0, 1 / 3, 0, 0, 0, 0],
                            [0, 0, 1 / 3, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]], dtype=complex)

gamma = 2 * sp.pi * 4800
decay_matrix = sp.asarray([[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, gamma, 0, 0],
                           [0, 0, 0, 0, gamma, 0],
                           [0, 0, 0, 0, 0, gamma]])

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
frequencies_p = sp.vstack((sp.hstack((sp.zeros((3, 3)), frequencies_intermediate)), sp.hstack((-frequencies_intermediate, sp.zeros((3, 3))))))

detunings_p = sp.linspace(-10 * gamma, 10 * gamma, 100)
dipole_operator_p = 0.063 * muB * sp.asarray([[0, 0, 0, sp.sqrt(0.03), sp.sqrt(0.22), sp.sqrt(0.75)],
                                              [0, 0, 0, sp.sqrt(0.12), sp.sqrt(0.68), sp.sqrt(0.2)],
                                              [0, 0, 0, sp.sqrt(0.85), sp.sqrt(0.1), sp.sqrt(0.05)],
                                              [sp.sqrt(0.03), sp.sqrt(0.12), sp.sqrt(0.85), 0, 0, 0],
                                              [sp.sqrt(0.22), sp.sqrt(0.68), sp.sqrt(0.10), 0, 0, 0],
                                              [sp.sqrt(0.75), sp.sqrt(0.2), sp.sqrt(0.05), 0, 0, 0]])

# frequencies_c = sp.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# dipole_operator_c = a0 * e_charge * sp.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

# Simulation parameters _________________________
dt = 1e-9
nt = 30000
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)

# Objects _____________________________________________________________________
the_atom = atom(initial_state, decay_matrix)
the_hamiltonian_p = hamiltonian_construct(dipole_operator_p, field_amplitude_p, frequencies_p)
# the_hamiltonian_c = hamiltonian_construct(dipole_operator_c, field_amplitude_c, frequencies_c)
the_simulation = simulation(the_atom, [the_hamiltonian_p], nt, dt)

# Run the simulation __________________________________________________________
t1 = time.time()
the_flop = the_simulation.time_evolution()
# the_susceptibility = the_simulation.susceptibility(detunings_p)
print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")

# Plot dat shit _______________________________________________________________
fig, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(the_times * 1e6, abs(the_flop[:, 0, 0]), label=r"$\rho_{11}$")
ax[0].plot(the_times * 1e6, abs(the_flop[:, 3, 3]), label=r"$\rho_{44}$")
ax[0].legend()

ax[1].plot(the_times * 1e6, the_flop[:, 0, 3].real, label=r"$\Re[\rho_{14}]$")
ax[1].plot(the_times * 1e6, the_flop[:, 0, 3].imag, label=r"$\Im[\rho_{14}]$")
ax[1].legend()

# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.plot(detunings_p / 1e6, the_susceptibility[:, 0, 2].real, label=r"$\Re[\chi]$")
# ax.plot(detunings_p / 1e6, 10 * the_susceptibility[:, 0, 2].imag, label=r"$10\times\Im[\chi]$")
# ax.legend()

# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.plot(detunings_p / 1e6, abs(the_susceptibility[:, 0, 0]), label=r"$\Re[\chi]$")
# ax.plot(detunings_p / 1e6, abs(the_susceptibility[:, 2, 2]), label=r"$\Im[\chi]$")
# ax.legend()

plt.show()

