import scipy as sp
import matplotlib.pyplot as plt
from density_matrix_classes.physicalconstants import *
from test_programs.Parameters_seven import *
file = "/home/zach/GitHub/Density-Matrix-Simulations/test_programs/Seven_Level/June_20_2018/9_21_46/"
data = sp.load(file + "data.txt.npy")
times = sp.load(file + "times.txt.npy")
print(0.063 * muB * ionic_density, sp.shape(data))
coherence = sp.zeros(sp.shape(data)[0], dtype=complex)
for i in range(3):
    for j in range(4, 7):
        coherence = coherence + data[:, j, i]
        print(i, j)
coherence = 0.063 * muB * (coherence) * 9.35e24 / 9

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(times / 1e6, abs(coherence))
ax.plot(times / 1e6, coherence.real)
ax.plot(times / 1e6, abs(coherence.real))
ax.plot(times / 1e6, coherence.imag)
ax.plot(times / 1e6, abs(coherence.imag))
plt.show()