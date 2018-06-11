import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from density_matrix_classes.physicalconstants import *

def Rabi_freq(dip, field):
    """
    
    :param dip: 
    :param field:
    :return: 
    """
    return field * dip / hbar

def rho_13(del_om, del_om_p, dip_p, dip_c, field_p, field_c, gamma):
    """
    
    :param del_om: 
    :param del_om_p: 
    :param dip_p: 
    :param dip_c: 
    :param field_p: 
    :param field_c: 
    :param gamma: 
    :return: 
    """
    rabi_p = Rabi_freq(dip_p, field_p)
    rabi_c = Rabi_freq(dip_c, field_c)
    coef = 2 * 9.35e24 * mu0 * (0.063 * muB)**2 / (hbar * rabi_p)
    return coef * 2 * del_om * rabi_p / (4 * del_om * del_om_p - rabi_c**2 + 2 * 1j * del_om * gamma)

gamma = 100 * 1 / 33e-6

n = 1.8
power_c = 8e-3
waist_c = 101e-6 / 2
intensity_c = power_c / (2 * sp.pi * waist_c**2)
# field_amplitude_c = sp.sqrt(2 / (n * c * epsilon0) * power_c / (sp.pi * waist_c**2))
dip_c = 0.063 * muB
field_amplitude_c = hbar * gamma / (dip_c * sp.sqrt(10))  #  sp.sqrt(4 * mu0 * n * power_c / (c * sp.pi * waist_c**2))
offset_c = 0

power_p = 0.75e-3
waist_p = 50e-6 / 2
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(4 * mu0 * n * power_p / (c * sp.pi * waist_p**2))
# field_amplitude_p = sp.sqrt(2 / (n * c * epsilon0) * power_p / (sp.pi * waist_p**2))
dip_p = 0.063 * muB
offset_p = 0  #offset_c + 1e6
print(gamma / 10, sp.sqrt(9/80) * gamma)

inhomogeneous = sp.linspace(-1e9, 1e9, 80000)

detunings = sp.linspace(-20e6, 20e6, 1000)
response_bare = rho_13(detunings, detunings + offset_c, dip_p, dip_c, field_amplitude_p, field_amplitude_c, gamma)
response = sp.zeros(1000, dtype=complex)
single_reponse = rho_13(detunings + offset_p, detunings + offset_c, dip_p, dip_c, field_amplitude_p, field_amplitude_c, gamma)
print()
# for i in inhomogeneous:
#         response = response + rho_13(detunings + i, detunings + offset_c, dip_p, dip_c, field_amplitude_p, field_amplitude_c, gamma)

response = response / len(inhomogeneous)
fig, ax = plt.subplots(nrows=1, ncols=1)
fig.subplots_adjust(hspace=0.5)
ax.plot(sp.asarray(detunings / 1e6), single_reponse.real, label="real")
ax.plot(sp.asarray(detunings / 1e6), single_reponse.imag, label="imag")
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(sp.sqrt(9/80) * gamma / 1e6, color='black', linewidth=0.5)
ax.axvline(-sp.sqrt(9/80) * gamma / 1e6, color='black', linewidth=0.5)
ax.set_xlabel("Detuning (MHz)")
ax.set_ylabel("Susceptibility")
ax.set_title(r"Single Atom Susceptibility ($\Omega_c$=" + str(round(field_amplitude_c * dip_c / hbar / 1e6, 3)) + " MHz)")
ax.legend()

# ax[1].plot(sp.asarray(detunings / 1e6), response.real, label="real")
# ax[1].plot(sp.asarray(detunings / 1e6), response.imag, label="imag")
# ax[1].set_xlabel("Detuning (kHz)")
# ax[1].set_ylabel("Amplitude (arb. units)")
# ax[1].axhline(0, color='black', linewidth=1)
# ax[1].axvline(0, color='black', linewidth=1)
# ax[1].set_title("Inhomogeneously Broadened Susceptibility")
# ax[1].legend()
# ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax[1].set_yticklabels([])
# ax[1].grid(which='both')
plt.show()
