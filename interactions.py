# class AL_interaction(object):
#
#     version = 1
#
#     # Turn on parameters for an ~80 MHz AOM
#     tau = 9e-9
#     t_naught = 90e-9
#
#     def __init__(self, field, dipoles, lower_detunings, upper_detunings, targets, detuning=0, shift=0):
#         """
#         Class For creating the interaction part of The Hamiltonian for the semiclassical atomic laser interacton
#         in the dipole approximation.  The unitary transformation that takes us to the interaction picture leaves
#         zeros along the diagonal of the Hamiltonian and oscillitory terms on the off diagonals.  This limits the
#         maximum frequency involved in the problem is on the order of 100 MHz instead of an optical frequency.
#
#         One instance per field.
#
#         Important note: A positive detuning means that the laser is tuned to a higher frequency than the transition.  A
#                         positive shift means that the levels have been shift to a higher frequency.  Therefore the shift
#                         and detuning need to have opposite signs to have the same effect.  If they have the same
#                         magnitude and sign they will cancel out.
#         :param field: Field amplitude.
#         :param dipoles: Dipole moment operator with elements dipoles_nm = <n|mu|m>.
#         :param lower_detunings: Frequency differences between the lower states.
#         :param upper_detunings: Frequency differences between the upper states.
#         :param targets: The states that the laser is roughly tuned to.
#         :param detuning: Frequency difference between the laser frequency and the frequency of the target transition.
#         :param shift: Frequency that the atomic states are shifted by resulting from any inhomogeneous broadening.
#                       Default is 0.
#         """
#         self.field = field
#         self.dipoles = dipoles
#         self.upper_detunings = upper_detunings
#         self.lower_detunings = lower_detunings
#         self.targets = targets
#         self.detuning = detuning  # laser detuning from resonance (not to useful for inhomogeneous broadening)
#         self.shift = shift  # shift of the upper levels from the lower due to the inhomogeneous broadening
#
#         self.Rabi = field * dipoles / hbar
#
#         self.n_upper = len(upper_detunings) + 1  # number of upper states
#         self.n_lower = len(lower_detunings) + 1  # number of lower states
#         self.n = len(upper_detunings) + len(lower_detunings) + 2  # total number of states
#
#         self.omega_p = self.generate_omega_p(targets)
#         self.unitary_frequencies = self.generate_unitary_frequencies()
#         self.interaction_frequencies = self.generate_interaction_frequencies()
#
#     def generate_omega_p(self, values):
#         """
#         Calculate the frequency of the interacting laser beam.  Since the frequency difference between
#         the highest ground state and lowest excited state is common to both the perturbing field and the
#         frequencies of the upper states, it will end up cancelling out in the interaction picture.
#
#         Not in the interaction picture.
#
#         :param values: Target state.
#         :return: frequency of the beam.
#         """
#         omega_p = sum(self.lower_detunings[values[0] - 1:self.n_lower - 1]) \
#                   + sum(self.upper_detunings[:values[1] - self.n_lower - 1]) + self.detuning
#         return omega_p
#
#     def generate_unitary_frequencies(self):
#         """
#         Calculate the frequencies to be used in the unitary transformation to the interaction picture.  The lowest
#         ground state is set to zero.  The frequency difference between the highest ground state and the lowest excited
#         state is not included either since we are working in the interaction frame.
#         :return: 1-D ndarray containing the frequencies.
#         """
#         temp = sp.hstack((0, self.lower_detunings, 0, self.upper_detunings))
#         temp[self.n_lower] += self.shift  # all the excited states will be shifted by self.shift
#         unitary_frequencies = sp.zeros(self.n)
#         for i in range(self.n):
#             unitary_frequencies[i] = sum(temp[:i+1])
#         return unitary_frequencies
#
#     def generate_interaction_frequencies(self):
#         """
#         Generate the frequencies for the upper right block matrix of the interaction Hamiltonian.
#         :return: A 2-D ndarray containing the frequencies.
#         """
#         u = sp.zeros((self.n_lower, self.n_upper))
#         for i in range(self.n_upper):
#             for j in range(self.n_lower):
#                 u[j, i] = self.omega_p + self.unitary_frequencies[j] - self.unitary_frequencies[i + self.n_lower]
#         return u
#
#     def U(self, t):
#         """
#         Unitary transformation to the interaction picture.
#         :param t: Time.
#         :return: Matrix representation of the transformation at time t as a complex ndarray.
#         """
#         return sp.diag(sp.exp(-1j * self.unitary_frequencies * t))
#
#     def U_dag(self, t):
#         """
#         Hermitian conjugate of the unitary transformation to the interaction picture.
#         :param t: Time.
#         :return: Matrix representation of the transformation at time t as a complex ndarray.
#         """
#         return sp.diag(sp.exp(1j * self.unitary_frequencies * t))
#
#     def oscillations(self, t):
#         """
#         Construct the oscilliatory part of the interaction Hamiltonian.
#         :param t: Time.
#         :return: 2-D complex ndarray whose elements are complex exponentials.
#         """
#         zeros1 = sp.zeros((self.n_lower, self.n_lower), dtype=complex)
#         zeros2 = sp.zeros((self.n_upper, self.n_upper), dtype=complex)
#         temp1 = sp.hstack((zeros1, sp.exp(1j * self.interaction_frequencies * t)))
#         temp2 = sp.hstack((sp.exp(1j * -self.interaction_frequencies.transpose() * t), zeros2))
#         ponents = sp.vstack((temp1, temp2))
#
#         return ponents
#
#     def sigmoid(self, t):
#         """
#         An offensive term for a particular type of function.  Used to model the turn on time of the laser beam.
#         :param t:
#         :return:
#         """
#         return 1 / (1 + sp.exp(-(t - AL_interaction.t_naught) / AL_interaction.tau))
#
#     def interaction(self, t):
#         """
#         Computes the interaction picture Hamiltonian at time t.
#         :param t: Time.
#         :return: Interaction Hamiltonian at time t as a complex ndarray.
#         """
#         ponents = self.oscillations(t)
#         return hbar * sp.multiply(self.Rabi, ponents)# * self.sigmoid(t)