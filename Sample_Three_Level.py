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
initial_state = sp.asarray([[1, 0], [0, 0]], dtype=complex)
gamma = 2 * sp.pi * 1e6
decay_matrix = sp.asarray([[0, 0], [0, gamma]])

# Beam parameters
power = 0.001
waist = 2000e-6
intensity = power / (2 * sp.pi * waist**2)
field_amplitude = sp.sqrt(2 * intensity / (n * epsilon0 * c))

# Interaction parameters
detuning = 0 * gamma / 2
frequencies = sp.asarray([[0, detuning], [-detuning, 0]])
detunings = sp.linspace(-1 / 5 * gamma, 1 / 5 * gamma, 100)
dipole_operator = a0 * e_charge * sp.asarray([[0, 1], [1, 0]])

# Simulation parameters
dt = 1e-9
nt = 1000
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)
print(a0 * e_charge / hbar, field_amplitude, field_amplitude * a0 * e_charge / hbar, 1 / (field_amplitude * a0 * e_charge / hbar) * 1e6)

# Objects
the_atom = atom(initial_state, decay_matrix)
the_hamiltonian = hamiltonian_construct(dipole_operator, field_amplitude, frequencies)
the_simulation = simulation(the_atom, the_hamiltonian.hamiltonian, nt, dt)

# Run the simulation
t1 = time.time()
# the_flop = the_simulation.time_evolution()
the_susceptibility = the_simulation.susceptibility([the_hamiltonian], detunings)
print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")

# Plot dat shit
# fig, ax = plt.subplots(nrows=2, ncols=1)
# ax[0].plot(the_times * 1e6, abs(the_flop[:, 0, 0]), label=r"$\rho_{11}$")
# ax[0].plot(the_times * 1e6, abs(the_flop[:, 1, 1]), label=r"$\rho_{22}$")
# ax[0].legend()
#
# ax[1].plot(the_times * 1e6, the_flop[:, 0, 1].real, label=r"$\Re[\rho_{12}]$")
# ax[1].plot(the_times * 1e6, the_flop[:, 0, 1].imag, label=r"$\Im[\rho_{12}]$")
# ax[1].legend()


fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(detunings / 1e6, the_susceptibility[:, 0, 1].real, label=r"$\Re[\chi]$")
ax.plot(detunings / 1e6, the_susceptibility[:, 0, 1].imag, label=r"$\Im[\chi]$")
ax.legend()

plt.show()

