from density_matrix_classes.physicalconstants import *
import scipy as sp

n_refraction = 1.8
initial_state =
decay_matrix =
decay_to =

number_of_atoms =
ib_linewidth =

# Beam parameters
power =
waist =
intensity = power / (2 * sp.pi * waist**2)
field_amplitude_p = sp.sqrt(2 * intensity / (n_refraction * epsilon0 * c))

# Interaction parameters
detuning =
frequencies =
detunings =
dipole_operator =
Rabi_f = a0 * e_charge * field_amplitude_p / hbar

# Simulation parameters
dt =
nt =
the_times = sp.linspace(0, nt * dt, nt, endpoint=False)