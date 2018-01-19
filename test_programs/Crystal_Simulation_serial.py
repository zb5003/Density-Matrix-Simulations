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
the_flop = sp.zeros((nt, *sp.shape(initial_state)), dtype=complex)

t1 = time.time()
for index in range(len(number_of_atoms)):
    if number_of_atoms[index] == 0:
        pass
    else:
        # Objects _____________________________________________________________________
        the_atom = atom(initial_state, decay_matrix, decay_to)
        the_hamiltonian_p = hamiltonian_construct(dipole_operator, field_amplitude, frequencies[index])
        single_simulation = single_atom_simulation(the_atom, [the_hamiltonian_p], nt, dt)
        the_simulation = inhomogeneous_broadening(single_simulation, ib_linewidth, number_of_atoms[index])

        # Run the simulation __________________________________________________________
        the_flop = the_flop + the_simulation.broadened_time_evolution()

print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")

the_flop = the_flop / sp.count_nonzero(number_of_atoms)

# Save dat shit _______________________________________________________________
populations_plot(the_times * 1e6, the_flop, loc)
crystal_pop_compare(the_times * 1e6, the_flop, loc)
coherence_plot(the_times * 1e6, the_flop, loc)
total_coherence_7(the_times * 1e6, the_flop, loc)
sp.save(loc + "/data.txt", the_flop)
sp.save(loc + "/times.txt", the_times)

