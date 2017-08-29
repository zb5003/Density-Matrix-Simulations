import datetime
import os
import scipy as sp
import matplotlib.pyplot as plt
from physicalconstants import *

def file_manager(subfolder):
    """
    Checks to see if a folder for that day's data exists.  If it does not exist, it is created.
    Folder format is month_day_year/subfolder/.
    :param subfolder: The name of the subfolder.  The format is power_value_length_value.
    :return: Folder.
    """
    months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May',
              6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October',
              11: 'November', 12: 'December'}
    now = datetime.datetime.now()
    folder_day = months[now.month] + "_" + str(now.day) + "_" + str(now.year)

    if not os.path.exists(subfolder + "/" + folder_day):
        os.makedirs(subfolder + "/" + folder_day)

    run = subfolder + "/" + folder_day + "/" + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)
    os.mkdir(run)

    return run

def populations_plot(times, density_m, location):
    """
    Plot the populations of each state versus time in separate files.
    :param times: The times over which to plot the populations in microseconds.
    :param density_m: The density matrix at each time step.
    :param location: Directory where plots will be saved.
    :return: None.
    """
    for i in range(sp.shape(density_m)[1]):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(times, abs(density_m[:, i, i]))
        ax.set_title(r"Population of State " + str(i + 1) + " v. Time")
        ax.set_xlabel(r"Time ($\mu$s)")
        ax.set_ylabel(r"$|\rho_{" + str(i + 1) + str(i + 1) + "}|$")
        ax.axhline(0, color='black')
        plt.savefig(location + "/State " + str(i + 1) + " population.png")
        plt.close()

    return None

def crystal_pop_compare(times, density_m, location):
    """
    Plot the populations of each state versus time in separate files.
    :param times: The times over which to plot the populations in microseconds.
    :param density_m: The density matrix at each time step.
    :param location: Directory where plots will be saved.
    :return: None. 
    """
    fig, ax = plt.subplots(nrows=3, ncols=1)
    fig.subplots_adjust(hspace=0.75)
    plt.suptitle("Population of Crystal Levels versus Time")
    ax[0].plot(times, abs(density_m[:, 0, 0]), label=r"$\rho_{11}$", color='red')
    ax[0].plot(times, abs(density_m[:, 3, 3]), label=r"$\rho_{44}$", color='black')
    ax[0].legend(bbox_to_anchor=(1, 1), loc=2)
    ax[0].set_xlabel(r"Time ($\mu$s)")
    ax[0].set_ylabel("Population")

    ax[1].plot(times, abs(density_m[:, 1, 1]), label=r"$\rho_{22}$", color='orange')
    ax[1].plot(times, abs(density_m[:, 2, 2]), label=r"$\rho_{33}$", color='yellow')
    ax[1].legend(bbox_to_anchor=(1, 1), loc=2)
    ax[1].set_xlabel(r"Time ($\mu$s)")
    ax[1].set_ylabel("Population")

    ax[2].plot(times, abs(density_m[:, 4, 4]), label=r"$\rho_{55}$", color='green')
    ax[2].plot(times, abs(density_m[:, 5, 5]), label=r"$\rho_{66}$", color='blue')
    ax[2].plot(times, abs(density_m[:, 6, 6]), label=r"$\rho_{77}$", color='purple')
    ax[2].legend(bbox_to_anchor=(1, 1), loc=2)
    ax[2].set_xlabel(r"Time ($\mu$s)")
    ax[2].set_ylabel("Population")

    plt.savefig(location + "/Crystal_Population_Comparison.png", bbox_inches="tight")
    plt.close()

    return None

def coherence_plot(times, density_m, location):
    """
    Plot the coherences versus time in separate files.
    :param times: The times over which to plot the populations in microseconds.
    :param density_m: The density matrix at each time step.
    :param location: Directory where plots will be saved.
    :return: None. 
    """
    for i in range(sp.shape(density_m)[1] - 1):
        for j in range(i + 1, sp.shape(density_m)[1]):
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(times, density_m[:, i, j].real, label=r'$\Re[\rho_{' + str(i + 1) + str(j + 1) + '}]$')
            # ax.plot(times, density_m[:, i, j].imag, label=r'$\Im[\rho_{' + str(i + 1) + str(j + 1) + '}]$')
            ax.set_title(r"Coherence Between States " + str(i + 1) + " and " + str(j + 1) + "v. Time")
            ax.set_xlabel(r"Time ($\mu$s)")
            ax.set_ylabel(r"$|\rho_{" + str(i + 1) + str(j + 1) + "}|$")
            ax.axhline(0, color='black')
            ax.legend(bbox_to_anchor=(1, 1), loc=2)
            plt.savefig(location + "/Coherence Between States " + str(i + 1) + " and " + str(j + 1) + ".png", bbox_inches="tight")
            plt.close()

    return None

def ground_v_excited_7(times, density_m, location):
    """
    Plot the dynamics of the sum of the three ground hyperfine state populations and the three excited hyperfine state
    populations.
    :param times: The times over which to plot the populations in microseconds.
    :param density_m: The density matrix at each time step.
    :param location: Directory where plots will be saved.
    :return: None. 
    """
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ground = abs(density_m[:, 0, 0]) + abs(density_m[:, 1, 1]) + abs(density_m[:, 2, 2])
    excited = abs(density_m[:, 4, 4]) + abs(density_m[:, 5, 5]) + abs(density_m[:, 6, 6])
    total = abs(sum([density_m[:, i, i] for i in range(7)]))
    fig.subplots_adjust(hspace=0.5)
    ax[0].plot(times, ground, label="Total Ground State")
    ax[0].plot(times, excited, label="Total Excited State")
    ax[0].set_title("Total Ground and Excited State v. Time")
    ax[0].set_xlabel(r"Time ($\mu$s)")
    ax[0].set_ylabel("State Population")
    ax[0].legend(bbox_to_anchor=(1, 1), loc=2)

    ax[1].plot(times, 1 + excited - ground)
    ax[1].set_title("Absoprtion v. Time")
    ax[1].set_xlabel(r"Time ($\mu$s)")
    ax[1].set_ylabel("Absorption (Arb. Units)")

    plt.savefig(location + "/Total_State_Population_Dynamics.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(times, total)
    ax.set_ylim([-0.1, 1.2])
    plt.savefig(location + "/Total_population.png")

    return None

def total_coherence_7(times, density_m, location):
    """
    Plot the sum of all the off diagonal elements (in the upper triangle) versus time.
    :param times: The times over which to plot the populations in microseconds.
    :param density_m: The density matrix at each time step.
    :param location: Directory where plots will be saved.
    :return: None. 
    """
    coherence = sp.zeros(sp.shape(density_m)[0], dtype=complex)
    for i in range(6):
        for j in range(i + 1, 7):
            coherence = coherence + density_m[:, i, j]
    coherence = 0.063 * muB * (coherence / 9) * 9.35e24

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    ax[0].plot(times, coherence.real, label=r"$\Re[\rho_{off-diagonal}]$")
    ax[0].plot(times, coherence.imag, label=r"$\Im[\rho_{off-diagonal}]$")
    ax[0].set_title("Real and Imaginary Components of Total Coherence v. Time")
    ax[0].set_xlabel(r"Time ($\mu$s)")
    ax[0].set_ylabel("Coherence")
    ax[0].legend(bbox_to_anchor=(1, 1), loc=2)

    ax[1].plot(times, abs(coherence))
    ax[1].set_title("Absolute Value of Total Coherence v. Time")
    ax[1].set_xlabel(r"Time ($\mu$s)")
    ax[1].set_ylabel("Coherence")

    plt.savefig(location + "/Total_Coherence.png", bbox_inches="tight")
    plt.close()

    return None

def ground_v_excited_2(times, density_m, location):
    """
    Plot the dynamics of the sum of the three ground hyperfine state populations and the three excited hyperfine state
    populations.
    :param times: The times over which to plot the populations in microseconds.
    :param density_m: The density matrix at each time step.
    :param location: Directory where plots will be saved.
    :return: None.
    """
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ground = abs(density_m[:, 0, 0])
    excited = abs(density_m[:, 1, 1])
    total = abs(sum([density_m[:, i, i] for i in range(2)]))
    fig.subplots_adjust(hspace=0.5)
    ax[0].plot(times, ground, label="Total Ground State")
    ax[0].plot(times, excited, label="Total Excited State")
    ax[0].set_title("Total Ground and Excited State v. Time")
    ax[0].set_xlabel(r"Time ($\mu$s)")
    ax[0].set_ylabel("State Population")
    ax[0].legend(bbox_to_anchor=(1, 1), loc=2)

    ax[1].plot(times, excited - ground)
    ax[1].set_title("Absoprtion v. Time")
    ax[1].set_xlabel(r"Time ($\mu$s)")
    ax[1].set_ylabel("Absorption (Arb. Units)")

    ax[2].plot(times, total)

    plt.savefig(location + "/Total_State_Population_Dynamics.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(times, total)
    ax.set_ylim([-0.1, 1.2])
    plt.savefig(location + "/Total_population.png")

    return None

def total_coherence_2(times, density_m, location):
    """
    Plot the sum of all the off diagonal elements (in the upper triangle) versus time.
    :param times: The times over which to plot the populations in microseconds.
    :param density_m: The density matrix at each time step.
    :param location: Directory where plots will be saved.
    :return: None.
    """
    coherence = sp.zeros(sp.shape(density_m)[0], dtype=complex)
    for i in range(2):
        for j in range(i + 1, 2):
            coherence = coherence + density_m[:, i, j]
    coherence = coherence

    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(hspace=0.5)
    ax[0].plot(times, coherence.real, label=r"$\Re[\rho_{off-diagonal}]$")
    ax[0].plot(times, coherence.imag, label=r"$\Im[\rho_{off-diagonal}]$")
    ax[0].set_title("Real and Imaginary Components of Total Coherence v. Time")
    ax[0].set_xlabel(r"Time ($\mu$s)")
    ax[0].set_ylabel("Coherence")
    ax[0].legend(bbox_to_anchor=(1, 1), loc=2)

    ax[1].plot(times, abs(coherence))
    ax[1].set_title("Absolute Value of Total Coherence v. Time")
    ax[1].set_xlabel(r"Time ($\mu$s)")
    ax[1].set_ylabel("Coherence")

    plt.savefig(location + "/Total_Coherence.png", bbox_inches="tight")
    plt.close()

    return None