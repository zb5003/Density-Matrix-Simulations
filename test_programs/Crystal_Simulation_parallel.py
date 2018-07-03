"""
All units are in SI.
"""
import test_programs.Parameters_seven as parameters
from density_matrix_classes.Atomic_Simulation_Classes import *
from density_matrix_classes.Data_Saving_Functions import *
from density_matrix_classes.physicalconstants import *
import scipy as sp
import time
import shutil
import multiprocessing as mp
from multiprocessing import Pool
import functools


def callback(result, accumulator):
    accumulator += result

def evolve(index, detuning):
    # Objects _____________________________________________________________________
    the_atom = atom(parameters.initial_state, parameters.decay_matrix, parameters.decay_to)
    the_hamiltonian_p = hamiltonian_construct(parameters.dipole_operator, parameters.field_amplitude, parameters.frequencies[index])
    single_simulation = single_atom_simulation(the_atom, [the_hamiltonian_p], parameters.nt, parameters.dt)
    run = single_simulation.time_evolution(detuning)
    return run#, run[-1]

if __name__ == "__main__":
    loc = file_manager(parameters.filename)
    shutil.copy("Parameters_seven.py", loc)
    the_flop = sp.zeros((parameters.nt, parameters.n_states, parameters.n_states), dtype=sp.complex128)
    cb = functools.partial(callback, accumulator=the_flop)

    t1 = time.time()
    for index in range(len(parameters.number_of_atoms)):
        if parameters.number_of_atoms[index] == 0:
            pass
        else:
            the_detunings = sp.linspace(-parameters.ib_linewidth / 2, parameters.ib_linewidth / 2, parameters.number_of_atoms[index])

            # Run the simulation __________________________________________________________
            with Pool(mp.cpu_count()) as p:
                for j in the_detunings:
                    p.apply_async(evolve, args=(index, j), callback=cb)
                p.close()
                p.join()

    t2 = time.time() - t1
    print("Total time elapsed = " + str(round(t2, 4)) + " seconds")

    the_flop = the_flop / parameters.n_total

    # Save dat shit _______________________________________________________________
    populations_plot(parameters.the_times * 1e6, the_flop, loc)
    # crystal_pop_compare(parameters.the_times * 1e6, the_flop, loc)
    coherence_plot(parameters.the_times * 1e6, the_flop, loc)
    total_coherence_7(parameters.the_times * 1e6, the_flop, loc)
    ground_v_excited_7(parameters.the_times * 1e6, the_flop, loc)
    sp.save(loc + "/data.txt", the_flop)
    sp.save(loc + "/times.txt", parameters.the_times)
