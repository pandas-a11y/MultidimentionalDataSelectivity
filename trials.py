# Attempt at reproducing spiking version of 'Implementation of a high-dim brain
# in spiking neurons'
# Exhibits no correlation between spiking rates or selectivity for any N dimensions

from brian2 import *
import numpy as np

# Seeds for random generator that produce reliably unique spike times for dataset

# seed(44532)
# seed(77732)
# seed(22386)
# seed(21617)
seed(1)

# Speeding up the simulation

defaultclock.dt = 1*ms


# Helping functions

def generate_dataset(N_dimensions, M_inputs):
    dataset = []
    for i in range(0, N_dimensions):
        result = []
        last_max = 0
        for j in range(M_inputs + 1):
            n = randint(last_max + 1, last_max + 100)
            if n > last_max:
                last_max = n + 1
            result.append(n)
        dataset.append(result)
    return dataset


def generate_weights(input_vectors, N_dimensions, M_inputs):
    target_input = [i % 100 for i in np.array(input_vectors[:, M_inputs]).flatten()]
    target_input = np.asmatrix(target_input).reshape(N_dimensions, 1)
    wstr = target_input / np.linalg.norm(target_input)
    waux = 0.01 * np.random.randint(low=1, high=100, size=(N_dimensions, 1))
    wort = waux - (np.transpose(wstr) * waux).item() * wstr
    wort = 0.005 * wort  # must be small
    w0 = wstr * (tht + eps) / np.linalg.norm(target_input) + wort
    return np.array(w0).flatten() * volt


def generate_indices(N_dimensions, M_inputs):
    indices = []
    for i in range(0, N_dimensions):
        indices.append([i for _ in range(M_inputs + 1)])
    return np.asarray(indices).flatten()


# Simulation parameters
N_dimensions = tuple(range(5, 21, 1))
M_inputs = 100
eps = 0.01
tht = 20
simulation_duration = 12 * second

# Neuron parameters
taum = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
eqs = Equations('''dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
                dge/dt = -ge / taue : 1''')


# Starting trials for different amount of dimensions
trial_spikes = []

for N_trial in N_dimensions:
    selecting_neuron = NeuronGroup(1, eqs, threshold='v > vt',
                                   reset='v = vr', refractory=5 * ms, method='euler')
    selecting_neuron.v = vr
    spikes_mon = SpikeMonitor(selecting_neuron)

    input_vectors = np.asmatrix(
            generate_dataset(N_trial, M_inputs))  # N arrays of M nonrepeating ints 0 to sim dur
    input_indices = generate_indices(N_trial, M_inputs)
    input_times = np.array(input_vectors).flatten()
    input_sets = SpikeGeneratorGroup(N_trial, input_indices, input_times * ms)

    synapses = Synapses(input_sets, selecting_neuron, model='w : volt', on_pre='v += w')
    synapses.connect()
    synapses.w[:, :] = generate_weights(input_vectors, N_trial, M_inputs)
    #print(synapses.w[0])

    run(simulation_duration)
    trial_spikes.append(spikes_mon.count[0])
    #print(trial_spikes[-1])
    #print(spikes_mon.t[-1])

# Plotting results
plot(N_dimensions, trial_spikes)
xlabel('Number of signal dimensions')
ylabel('Number of spikes during a simulation')
tight_layout()
show()
