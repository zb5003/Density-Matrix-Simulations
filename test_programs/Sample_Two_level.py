"""
All units are in SI.
"""
import scipy as sp
import time
import matplotlib.pyplot as plt
import shutil
from density_matrix_classes.Atomic_Simulation_Classes import *
from density_matrix_classes.physicalconstants import *
from density_matrix_classes.Data_Saving_Functions import *
from density_matrix_classes.Inhomogeneous_Broadening_Classes import *

def analytic(Rabi, detuning, t):
    Rabi = Rabi
    general = sp.sqrt(detuning**2 + Rabi**2)
    real = -(detuning * Rabi) / (general**2) * sp.sin(general / 2 * t)**2
    imaginary = -1j * Rabi / general * sp.cos(general / 2 * t) * sp.sin(general / 2 * t)
    return (real + imaginary)# * sp.exp(1j * detuning * t)

# Atomic parameters
n = 1.8
initial_state = sp.asarray([[1, 0], [0, 0]], dtype=complex)
gamma = 2 * sp.pi * 1e6
decay_matrix = sp.asarray([[0, 0], [0, gamma]])
decay_to = sp.asarray([[0, sp.sqrt(gamma)], [0, 0]])

number_of_atoms = 400
ib_linewidth = 2 * sp.pi * 1e9

# Beam parameters
power_p = 0.0001
waist_p = 2000e-6
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(2 * intensity_p / (n * epsilon0 * c))

power_c = 0 * 0.0005
waist_c = 2000e-6
intensity_c = power_c / (2 * sp.pi * waist_c**2)
field_amplitude_c = sp.sqrt(2 * intensity_c / (n * epsilon0 * c))

# Interaction parameters
detuning = 5 * 1e6
frequencies_p = sp.asarray([[0, detuning], [-detuning, 0]])
detunings_p = sp.linspace(-20 * gamma, 20 * gamma, 100)
dipole_operator_p = a0 * e_charge * sp.asarray([[0, 1], [1, 0]])
Rabi_f = a0 * e_charge * field_amplitude_p / hbar

# Simulation parameters
dt = 1e-9
nt = 3000
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)

# Objects
the_atom = atom(initial_state, decay_matrix, decay_to)
the_hamiltonian_p = hamiltonian_construct(dipole_operator_p, field_amplitude_p, frequencies_p)
the_simulation = single_atom_simulation(the_atom, [the_hamiltonian_p], nt, dt)
inhomogeneously_broadened_simulation = inhomogeneous_broadening(the_simulation, ib_linewidth, number_of_atoms)

# Run the simulation
t1 = time.time()
# the_flop = inhomogeneously_broadened_simulation.broadened_time_evolution()
# the_flop = the_simulation.time_evolution(detuning)
the_susceptibility = the_simulation.susceptibility(detunings_p)
print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")

anal_sol = analytic(Rabi_f, 2 * detuning, the_times)

# Save dat shit _______________________________________________________________
# loc = file_manager("Two_Level_Inhomogeneously_Broadened")
# populations_plot(the_times * 1e6, the_flop, loc)
# # crystal_pop_compare(the_times * 1e6, the_flop, loc)
# coherence_plot(the_times * 1e6, the_flop, loc)
# ground_v_excited_2(the_times * 1e6, the_flop, loc)
# total_coherence_2(the_times * 1e6, the_flop, loc)
# shutil.copy("Crystal_Simulation.py", loc)
# Plot dat shit
# fig, ax = plt.subplots(nrows=3, ncols=1)
# ax[0].plot(the_times * 1e6, abs(the_flop[:, 0, 0]), label=r"$\rho_{11}$")
# ax[0].plot(the_times * 1e6, abs(the_flop[:, 1, 1]), label=r"$\rho_{22}$")
# ax[0].legend()
#
# ax[1].plot(the_times * 1e6, the_flop[:, 0, 1].real, label=r"$\Re[\rho_{12}]$")
# ax[1].plot(the_times * 1e6, the_flop[:, 0, 1].imag, label=r"$\Im[\rho_{12}]$")
# ax[1].legend()
#
# ax[2].plot(the_times * 1e6, anal_sol.real, label=r"$\Re[\rho_{12}]$")
# ax[2].plot(the_times * 1e6, anal_sol.imag, label=r"$\Im[\rho_{12}]$")
# ax[2].legend()
#
# ax[0].plot(the_times * 1e6, abs(the_flop[:, 0, 0]) + abs(the_flop[:, 1, 1]))

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(detunings_p / 1e6, the_susceptibility[:, 0, 1].real, label=r"$\Re[\chi]$")
ax.plot(detunings_p / 1e6, 5 * the_susceptibility[:, 0, 1].imag, label=r"$5\times\Im[\chi]$")
ax.legend()

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(detunings_p / 1e6, abs(the_susceptibility[:, 0, 0]), label=r"$\Re[\chi]$")
ax.plot(detunings_p / 1e6, abs(the_susceptibility[:, 1, 1]), label=r"$\Im[\chi]$")
ax.legend()

plt.show()

