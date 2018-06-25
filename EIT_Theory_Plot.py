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

def rho_13_full(del_om, del_om_p, dip_p, dip_c, field_p, field_c, gamma21, gamma31):
    """
    EIT theoretical response with dephasing.
    :param del_om: 
    :param del_om_p: 
    :param dip_p: 
    :param dip_c: 
    :param field_p: 
    :param field_c: 
    :param gamma21: 
    :param gamma31: 
    :return: 
    """
    rabi_p = Rabi_freq(dip_p, field_p)
    rabi_c = Rabi_freq(dip_c, field_c)
    coef = 2 * 9.35e24 * mu0 * (0.063 * muB)**2 / (hbar * rabi_p)
    numerator = rabi_p * (2 * del_om - 1j * gamma21)
    denominator = (4 * del_om * del_om_p - gamma21 * gamma31 - rabi_c**2) - 2 * 1j * (gamma21 * del_om_p + del_om * gamma31)
    return numerator / denominator

gamma = 1 / 33e-6
gamma21 = 5e4
gamma31 = 2e6
lambd = 527e-9

n = 1.8
power_c = 60e-3
waist_c = 112e-6 / 2
intensity_c = power_c / (2 * sp.pi * waist_c**2)
field_amplitude_c = sp.sqrt(4 * mu0 * n * power_c / (c * sp.pi * waist_c**2))
dip_c = 0.063 * muB
# field_amplitude_c = hbar * gamma / (dip_c * sp.sqrt(10))  #  sp.sqrt(4 * mu0 * n * power_c / (c * sp.pi * waist_c**2))
offset_c = 0

power_p = 150e-6
waist_p = 50e-6 / 2
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(4 * mu0 * n * power_p / (c * sp.pi * waist_p**2))
# field_amplitude_p = sp.sqrt(2 / (n * c * epsilon0) * power_p / (sp.pi * waist_p**2))
dip_p = 0.063 * muB
offset_p = 0  #offset_c + 1e6

inhomogeneous = sp.linspace(-1e9, 1e9, 80000)

detunings = sp.linspace(-5e6, 5e6, 50000)
response_bare = rho_13(detunings, detunings + offset_c, dip_p, dip_c, field_amplitude_p, field_amplitude_c, gamma)
response = sp.zeros(50000, dtype=complex)
single_reponse = rho_13(detunings + offset_c, detunings + offset_p, dip_p, dip_c, field_amplitude_p, field_amplitude_c, gamma)
single_response_full = rho_13_full(detunings + offset_c, detunings + offset_p, dip_p, dip_c, field_amplitude_p, field_amplitude_c, gamma21, gamma31)

# for i in inhomogeneous:
#         response = response + rho_13(detunings + offset_c, detunings + i, dip_p, dip_c, field_amplitude_p, field_amplitude_c, gamma)

response = response / len(inhomogeneous)
fig, ax = plt.subplots(nrows=2, ncols=1)
fig.subplots_adjust(hspace=0.5)
# ax[0].plot(sp.asarray(detunings / 1e6), single_reponse.real, label="real")
ax[0].plot(sp.asarray(detunings / 1e6), single_reponse.imag, label="imag")
ax[0].plot(sp.asarray(detunings / 1e6), sp.exp(-4 * sp.pi / lambd * single_reponse.imag / 2 * 0.01 / 1e4), label="imag")
ax[0].axhline(0, color='black', linewidth=0.5)
ax[0].axvline(sp.sqrt(9/80) * gamma / 1e6, color='black', linewidth=0.5)
ax[0].axvline(-sp.sqrt(9/80) * gamma / 1e6, color='black', linewidth=0.5)
ax[0].set_xlabel("Detuning (MHz)")
ax[0].set_ylabel("Susceptibility")
ax[0].set_title(r"Single Atom Susceptibility ($\Omega_c$=" + str(round(field_amplitude_c * dip_c / hbar / 1e6, 3)) + " MHz)")
ax[0].legend()

# ax[1].plot(sp.asarray(detunings / 1e6), response.real, label="real")
# ax[1].plot(sp.asarray(detunings / 1e6), response.imag, label="imag")
# ax[1].plot(sp.asarray(detunings / 1e6), single_response_full.real, label="real")
ax[1].plot(sp.asarray(detunings / 1e6), single_response_full.imag, label="imag")
ax[1].plot(sp.asarray(detunings / 1e6), sp.exp(-4 * sp.pi / lambd * single_response_full.imag / 2 * 0.01 / (1.6e9 / gamma31)), label="Transmission")
ax[1].set_xlabel("Detuning (MHz)")
ax[1].set_ylabel("Amplitude (arb. units)")
ax[1].axhline(0, color='black', linewidth=1)
ax[1].axvline(0, color='black', linewidth=1)
ax[1].set_title("Inhomogeneously Broadened Susceptibility")
ax[1].legend()
# ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax[1].set_yticklabels([])
ax[1].grid(which='both')
plt.show()
