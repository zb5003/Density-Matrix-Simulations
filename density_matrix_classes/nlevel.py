import scipy as sp
from density_matrix_classes.physicalconstants import *

def commutator(M1, M2):
    """
    Calculate the commutator [M1, M2].
    
    :param M1: Ndarray (possible complex). First matrix.
    :param M2: Ndarray (possible complex). Second matrix.
    :return: Ndarray (possible complex). The commutator between M1 and M2 as a 2-D ndarray.
    """
    return sp.dot(M1, M2) - sp.dot(M2, M1)

def anticommutator(M1, M2):
    """
    Calculate the anticommutator {M1, M2}.
    
    :param M1: Ndarray (possible complex). First matrix.
    :param M2: Ndarray (possible complex). Second matrix.
    :return: Ndarray (possible complex). The anticommutator between M1 and M2 as a 2-D ndarray.
    """
    return sp.dot(M1, M2) + sp.dot(M2, M1)

def rho_dot(Hamiltonian, Gamma, rho, closed):
    """
    Calculate the time derivative of the density matrix

    This function uses the Lindblad form of the quantum master equation (Steck eq. 5.177 & 5.178 pg 181)

    .. math::
        \\dot{\\rho} = -\\frac{i}{\\hbar} [H, \\rho] + \\Gamma D[c] \\rho

    where

    .. math::
        D[c] \\rho = c\\rho c^{\\dagger} - \\frac{1}{2} \\{c^{\\dagger} c, \\rho\\}.

    :math:`\Gamma` is the decay rate and :math:`c` along with its Hermitian conjugate :math:`c^{\dagger}` are the transition matrices
    of the n level system.
    The documentation also has some extra information.
    For now only radiative decay is considered (no dephasing).
    
    :param Hamiltonian: Complex ndarray. Hamiltonian.
    :param Gamma: Ndarray. Decay matrix.
    :param rho: Complex ndarray. density matrix.
    :return: Complex ndarray. Time derivative of the density matrix.
    """
    return -1j / hbar * commutator(Hamiltonian, rho) \
           - 1 / 2 * anticommutator(Gamma, rho) \
           + sum(sp.matmul(closed[0], sp.matmul(rho, closed[1])))

def RK_rho(Hamiltonian, Gamma, rho, closed, dt):
    """
    Calculates a single time step using the fourth order Runge-Kutta method.
    
    :param Hamiltonian: Complex array. Hamiltonian.
    :param Gamma: Array. Decay matrix.
    :param rho: Complex array. Density matrx.
    :param dt: Float. Time step.
    :return: Complex array. Density matrix after evolving for time dt.
    """
    F1 = dt * rho_dot(Hamiltonian, Gamma, rho, closed)
    F2 = dt * rho_dot(Hamiltonian, Gamma, rho + 1 / 2 * F1, closed)
    F3 = dt * rho_dot(Hamiltonian, Gamma, rho + 1 / 2 * F2, closed)
    F4 = dt * rho_dot(Hamiltonian, Gamma, rho + F3, closed)
    return rho + 1 / 6 * (F1 + 2 * F2 + 2 * F3 + F4)

def time_evolve(Hamiltonian, Gamma, rho, closed, dt, nt):
    """
    Perform multiple time step evolutions starting with the initial density matrix self.Dens_i.
    
    :param Hamiltonian: Complex array. Hamiltonian.
    :param dt: Float. Time step.
    :param nt: Int. Number of time steps.
    :return: Complex array. Density matrix after evolving for many time steps.
    """
    for i in range(nt):
        temp = RK_rho(Hamiltonian(i * dt), Gamma, rho, closed, dt)
        rho = temp
    return rho
