"""
All units are in SI.
"""
import scipy as sp
import time
import matplotlib.pyplot as plt
from density_matrix_classes.Atomic_Simulation_Classes import *
from density_matrix_classes.physicalconstants import *
from test_programs.Parameters_three import *

# print(Rabi_p, Rabi_c)

# Objects
the_atom = atom(initial_state, decay_matrix, decay_to)
the_hamiltonian_p = hamiltonian_construct(dipole_operator_p, field_amplitude_p, frequencies_p)
the_hamiltonian_c = hamiltonian_construct(dipole_operator_c, field_amplitude_c, frequencies_c)
the_simulation = single_atom_simulation(the_atom, [the_hamiltonian_p, the_hamiltonian_c], nt, dt)

# Run the simulation
t1 = time.time()
the_flop = the_simulation.time_evolution()
the_susceptibility = the_simulation.susceptibility(detunings_p)
print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")

# Plot dat shit
fig, ax = plt.subplots(nrows=3, ncols=1)
ax[0].plot(the_times * 1e6, abs(the_flop[:, 0, 0]), label=r"$\rho_{11}$")
ax[0].plot(the_times * 1e6, abs(the_flop[:, 1, 1]), label=r"$\rho_{22}$")
ax[0].plot(the_times * 1e6, abs(the_flop[:, 2, 2]), label=r"$\rho_{33}$")
ax[0].legend()

ax[1].plot(the_times * 1e6, the_flop[:, 0, 2].real, label=r"$\Re[\rho_{13}]$")
ax[1].plot(the_times * 1e6, the_flop[:, 0, 2].imag, label=r"$\Im[\rho_{13}]$")
ax[1].legend()

ax[2].plot(the_times * 1e6, the_flop[:, 0, 1].real, label=r"$\Re[\rho_{12}]$")
ax[2].plot(the_times * 1e6, the_flop[:, 0, 1].imag, label=r"$\Im[\rho_{12}]$")
ax[2].legend()

fig1, ax = plt.subplots(nrows=2, ncols=1)
ax[0].plot(detunings_p / 1e6, the_susceptibility[:, 0, 2].real, label=r"$\Re[\chi]$")
ax[0].axhline(0)
ax[0].axvline(0)
ax[1].plot(detunings_p / 1e6, -the_susceptibility[:, 0, 2].imag, label=r"$Im[\chi]$")
ax[1].axhline(0)
ax[1].axvline(0)
plt.legend()

plt.show()

