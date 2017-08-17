"""
All units are in SI.
"""
import scipy as sp
import time
import matplotlib.pyplot as plt
from Atomic_Simulation_Classes import *
from physicalconstants import *

# Atomic parameters
n = 1.8
initial_state = sp.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
gamma = 2 * sp.pi * 2e6
decay_matrix = sp.asarray([[0, 0, 0], [0, 0, 0], [0, 0, gamma]])

# Beam parameters
power_p = 0.01
waist_p = 2000e-6
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(2 * intensity_p / (n * epsilon0 * c))

power_c = 0.05
waist_c = 2000e-6
intensity_c = power_c / (2 * sp.pi * waist_c**2)
field_amplitude_c = sp.sqrt(2 * intensity_c / (n * epsilon0 * c))

# Interaction parameters
detuning = 0
frequencies_p = sp.asarray([[0, 0, detuning], [0, 0, 0], [-detuning, 0, 0]])
detunings_p = sp.linspace(-10 * gamma, 10 * gamma, 100)
dipole_operator_p = a0 * e_charge * sp.asarray([[0, 0, 1], [0, 0, 0], [1, 0, 0]])

frequencies_c = sp.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
dipole_operator_c = a0 * e_charge * sp.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0]])

# Simulation parameters
dt = 1e-9
nt = 3000
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)

# Objects
the_atom = atom(initial_state, decay_matrix)
the_hamiltonian_p = hamiltonian_construct(dipole_operator_p, field_amplitude_p, frequencies_p)
the_hamiltonian_c = hamiltonian_construct(dipole_operator_c, field_amplitude_c, frequencies_c)
the_simulation = single_atom_simulation(the_atom, [the_hamiltonian_p, the_hamiltonian_c], nt, dt)

# Run the simulation
t1 = time.time()
# the_flop = the_simulation.time_evolution()
the_susceptibility = the_simulation.susceptibility(detunings_p)
print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")

# Plot dat shit
# fig, ax = plt.subplots(nrows=2, ncols=1)
# ax[0].plot(the_times * 1e6, abs(the_flop[:, 0, 0]), label=r"$\rho_{11}$")
# ax[0].plot(the_times * 1e6, abs(the_flop[:, 2, 2]), label=r"$\rho_{22}$")
# ax[0].legend()
#
# ax[1].plot(the_times * 1e6, the_flop[:, 0, 2].real, label=r"$\Re[\rho_{12}]$")
# ax[1].plot(the_times * 1e6, the_flop[:, 0, 2].imag, label=r"$\Im[\rho_{12}]$")
# ax[1].legend()

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(detunings_p / 1e6, the_susceptibility[:, 0, 2].real, label=r"$\Re[\chi]$")
ax.plot(detunings_p / 1e6, 10 * the_susceptibility[:, 0, 2].imag, label=r"$10\times\Im[\chi]$")
ax.legend()

# fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.plot(detunings_p / 1e6, abs(the_susceptibility[:, 0, 0]), label=r"$\Re[\chi]$")
# ax.plot(detunings_p / 1e6, abs(the_susceptibility[:, 2, 2]), label=r"$\Im[\chi]$")
# ax.legend()

plt.show()

