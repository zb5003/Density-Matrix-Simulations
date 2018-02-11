import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import density_matrix_classes.physicalconstants as pc
import test_programs.Parameters as param

def burned_back(freq):
    sigma = 500e3
    return 1 / sp.sqrt(2 * sp.pi * sigma**2) * sp.exp(-freq**2 / (2 * sigma**2))

def Rabi_freq(dip, field):
    """
    
    :param dip: 
    :param field:
    :return: 
    """
    return field * dip / pc.hbar

def rho_13(del_om, del_om_p, rabi_c):
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
    coef = 2 * param.ionic_density * pc.mu0 * 0.2 * (0.063 * pc.muB)**2 / pc.hbar
    return coef * del_om / (4 * del_om * del_om_p - rabi_c**2 + 2 * 1j * del_om * param.gamma)

n = 1.8
power_c = 800e-3
waist_c = 101e-6 / 2
intensity_c = power_c / (2 * sp.pi * waist_c**2)
field_amplitude_c = sp.sqrt(4 * pc.mu0 * n * power_c / (pc.c * sp.pi * waist_c**2))
dip_c = 0.063 * sp.sqrt(0.05) * pc.muB
offset_c = 0e6
rabi_c = Rabi_freq(dip_c, field_amplitude_c)
print("Coupling intensity =", intensity_c / (100)**2, "W/cm^2")
print("Coupling Rabi frequency =", rabi_c / 1e6, "MHz")

power_p = 7.5e-6
waist_p = 50e-6 / 2
intensity_p = power_p / (2 * sp.pi * waist_p**2)
field_amplitude_p = sp.sqrt(4 * pc.mu0 * n * power_p / (pc.c * sp.pi * waist_p**2))
dip_p = 0.063 * sp.sqrt(0.2) * pc.muB
rabi_p = Rabi_freq(dip_p, field_amplitude_p)
print("Probe intensity =", intensity_p / (100)**2, "W/cm^2")
print("Probe Rabi frequency =", rabi_p / 1e6, "MHz")
print("Ratio of coupling Rabi frequency to probe Rabi frequency =", rabi_c / rabi_p)

trough = 10e6
peak = sp.linspace(-trough, trough, int(2 * trough / 25e3))
print(int(2 * trough / 25e3))
inhomogeneous = sp.hstack((sp.linspace(-1e9, -trough, 40000), sp.linspace(trough, 1e9, 40000), burned_back(peak)))
# inhomogeneous = sp.linspace(-1e9, 1e9, 80000)
# inhomogeneous = sp.hstack((sp.linspace(-1e9, -trough, 40000), sp.linspace(trough, 1e9, 40000), 0))
detunings = sp.linspace(-20e6, 20e6, 6000)
response_bare = rho_13(detunings + offset_c, detunings + offset_c, rabi_c)
response = sp.zeros(6000, dtype=complex)
single_reponse = rho_13(detunings + offset_c, detunings + offset_c, rabi_c)

for i in inhomogeneous:
        response = response + rho_13(detunings + offset_c + i, detunings + offset_c + i, rabi_c)

response = response / len(inhomogeneous)
fig, ax = plt.subplots(nrows=3, ncols=1)
fig.subplots_adjust(hspace=1)
ax[0].plot(detunings / 1e6, single_reponse.real, label="real")
ax[0].plot(detunings / 1e6, single_reponse.imag, label="imag")
ax[0].axhline(0, color='black', linewidth=0.5)
ax[0].axvline(0, color='black', linewidth=0.5)
ax[0].set_xlabel("Detuning (MHz)")
ax[0].set_ylabel("Susceptibility")
ax[0].set_title("Single Atom Susceptibility")
ax[0].legend()

ax[1].plot(detunings / 1e6, response.real, label="real")
ax[1].plot(detunings / 1e6, response.imag, label="imag")
ax[1].set_xlabel("Detuning (kHz)")
ax[1].set_ylabel("Amplitude (arb. units)")
ax[1].axhline(0, color='black', linewidth=1)
ax[1].axvline(0, color='black', linewidth=1)
ax[1].set_title("Inhomogeneously Broadened Susceptibility")
ax[1].legend()

ax[2].plot(detunings / 1e6, sp.exp(2 * sp.pi * response.imag * 0.1 / 527e-9))
ax[2].set_xlabel("Detuning (kHz)")
ax[2].set_ylabel("Transmission (%)")
ax[2].axhline(0, color='black', linewidth=1)
ax[2].axvline(0, color='black', linewidth=1)
ax[2].set_title("Inhomogeneously Broadened Absorption")
plt.show()
