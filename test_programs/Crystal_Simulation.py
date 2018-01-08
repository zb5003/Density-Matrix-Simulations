"""
All units are in SI.
"""
import scipy as sp
import time
import matplotlib.pyplot as plt
import shutil
from test_programs.Parameters import *
from density_matrix_classes.Atomic_Simulation_Classes import *
from density_matrix_classes.Data_Saving_Functions import *
from density_matrix_classes.physicalconstants import *

# Objects _____________________________________________________________________
the_atom = atom(initial_state, decay_matrix, decay_to)
the_hamiltonian_p = hamiltonian_construct(dipole_operator, field_amplitude, frequencies)
the_simulation = single_atom_simulation(the_atom, [the_hamiltonian_p], nt, dt)

# Run the simulation __________________________________________________________
t1 = time.time()
the_flop = the_simulation.time_evolution_serial()
print("Time elapsed = " + str(round(time.time() - t1, 4)) + " seconds")
print(frequencies)
print(frequency_matrix_generator(detuning, lower_spacing, upper_spacing))
# Save dat shit _______________________________________________________________
loc = file_manager(filename)
populations_plot(the_times * 1e6, the_flop, loc)
crystal_pop_compare(the_times * 1e6, the_flop, loc)
coherence_plot(the_times * 1e6, the_flop, loc)
shutil.copy("Parameters.py", loc)
sp.save(loc + "/data.txt", the_flop)
sp.save(loc + "/times.txt", the_times)

