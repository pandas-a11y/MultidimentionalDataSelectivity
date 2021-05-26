"""
This is a Python 3.6 version of script for Milena

Project with S. Lobov. Implementation of a high-dim brain
in spiking neurons.

The script illustrates the paper:
Tyukin et al., Bull. Math. Biol. 2018.

Try to execute this script several times for different
values of the dimension n
%
For example:
for n = 5 practically all the time you'll get a not-selective neuron
for n = 10 about half times you'll get a selective neuron
for n = 40 practically all the time you'll get a selective neuron
"""
import numpy as np

#Parameters
M = 500     # number of background stimuli
n = 20      # dimension of stimuli
tht = 0.7   # firing threshold
eps = 0.01  # constant

#Stimuli generation
#1. a hypersphere 2. a hypercube
s = np.matrix(2 * np.random.rand(n, M+1) - 1) #Hypercube

#Weight calculation
wstr = s[:,M] / np.linalg.norm(s[:,M])
waux = 0.01 * (2 * np.random.rand(n, 1) - 1)
wort = waux - (np.transpose(wstr) * waux).item() * wstr
wort = 0.005*wort # must be small
w = wstr * (tht + eps) / np.linalg.norm(s[:,M]) + wort

#Generating binary vector of response
u = np.transpose(w) * s - tht > 0

#Cheching response selectivity
result = np.where(u)[True]

if result[0] == M:
    print("The neuron is selective.")
else:
    print("The neuron responds to background stimuli at index: ")
    print(*result, sep='\n')
