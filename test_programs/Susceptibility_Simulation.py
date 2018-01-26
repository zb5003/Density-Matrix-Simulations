"""
All units are in SI.
"""
import scipy as sp
import time
import matplotlib.pyplot as plt
import shutil
from test_programs.Parameters import *
from density_matrix_classes.Inhomogeneous_Broadening_Classes import *
from density_matrix_classes.Atomic_Simulation_Classes import *
from density_matrix_classes.Data_Saving_Functions import *
from density_matrix_classes.physicalconstants import *

loc = file_manager(filename)
shutil.copy("Parameters.py", loc)
n_detunings = 100
n_linwidths = 2
the_flop = sp.zeros((n_detunings, *sp.shape(initial_state)), dtype=complex)
detunings = sp.linspace(-n_linwidths * gamma, n_linwidths * gamma, n_detunings)

t1 = time.time()
for index, i in enumerate(frequencies):
    # Objects _____________________________________________________________________
    the_atom = atom(initial_state, decay_matrix, decay_to)
    the_hamiltonian_p = hamiltonian_construct(dipole_operator, field_amplitude, i)
    single_simulation = single_atom_simulation(the_atom, [the_hamiltonian_p], nt, dt)

    # Run the simulation __________________________________________________________
    the_flop[index, :, :] = the_flop[index, :, :] + single_simulation.susceptibility(detunings)

print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")
plt.plot(detunings, the_flop[0, 4].real)
plt.plot(detunings, the_flop[0, 4].imag)
plt.show()

# the_flop = the_flop / sp.count_nonzero(number_of_atoms)
#
# # Save dat shit _______________________________________________________________
# populations_plot(the_times * 1e6, the_flop, loc)
# crystal_pop_compare(the_times * 1e6, the_flop, loc)
# coherence_plot(the_times * 1e6, the_flop, loc)
# total_coherence_7(the_times * 1e6, the_flop, loc)
# sp.save(loc + "/data.txt", the_flop)
# sp.save(loc + "/times.txt", the_times)

