import scipy as sp
from density_matrix_classes.physicalconstants import *

def commutator(M1, M2):
    """
    Calculates the commutator [M1, M2].
    :param M1: First matrix.
    :param M2: Second matrix.
    :return: The commutator between M1 and M2 as a 2-D ndarray.
    """
    return sp.dot(M1, M2) - sp.dot(M2, M1)

def anticommutator(M1, M2):
    """
    Calculates the anticommutator {M1, M2}.
    :param M1: First matrix.
    :param M2: Second matrix.
    :return: The anticommutator between M1 and M2 as a 2-D ndarray.
    """
    return sp.dot(M1, M2) + sp.dot(M2, M1)

def rho_dot(Hamiltonian, Gamma, rho, closed):
    """
    Calculate the time derivative of the density matrix using the quantum master equation (Steck eq. 5.177 & 5.178 pg 181)
        rho_dot = -i / hbar * [H, rho] + Gamma * D[c] * rho
    where
        D[c] * rho = 1 / 2 * {c_dagger * c, rho}.
    Gamma is the decay rate and c are the transition matrices of the n level system. In practice c will be a sum of 
    multiple transition matrices.
    
    NOTE: 1.) For now only radiative decay is considered (no dephasing) and all levels have the same decay rate.
          2.) The Lindblad operator does not include the term that adds population back in.  Therefore this function 
              simulates open quantum systems
    
    :param Hamiltonian: Hamiltonian.
    :param Gamma: Decay matrix.
    :param rho: density matrix.
    :return: Time derivative of the density matrix as a 2-D ndarray (dtype=complex).
    """
    return -1j / hbar * commutator(Hamiltonian, rho) \
           - 1 / 2 * anticommutator(Gamma, rho) \
           + sum(sp.matmul(closed[0], sp.matmul(rho, closed[1])))

def RK_rho(Hamiltonian, Gamma, rho, closed, dt):
    """
    Calculates a single time step using the fourth order Runge-Kutta method.
    :param Hamiltonian: Hamiltonian.
    :param Gamma: Decay matrix.
    :param rho: Density matrx.
    :param dt: Time step.
    :return: Density matrix after evolving for time dt as a 2-D ndarray (dtype=complex).
    """
    F1 = dt * rho_dot(Hamiltonian, Gamma, rho, closed)
    F2 = dt * rho_dot(Hamiltonian, Gamma, rho + 1 / 2 * F1, closed)
    F3 = dt * rho_dot(Hamiltonian, Gamma, rho + 1 / 2 * F2, closed)
    F4 = dt * rho_dot(Hamiltonian, Gamma, rho + F3, closed)
    return rho + 1 / 6 * (F1 + 2 * F2 + 2 * F3 + F4)

def time_evolve(Hamiltonian, Gamma, rho, closed, dt, nt):
    """
    Perform multiple time step evolutions starting with the initial density matrix self.Dens_i.
    :param Hamiltonian: Hamiltonian.
    :param dt: Time step.
    :param nt: Number of time steps.
    :return: Density matrix after evolving for many time steps as a 2-D ndarray (dtype=complex).
    """
    for i in range(nt):
        temp = RK_rho(Hamiltonian(i * dt), Gamma, rho, closed, dt)
        rho = temp
    return rho
