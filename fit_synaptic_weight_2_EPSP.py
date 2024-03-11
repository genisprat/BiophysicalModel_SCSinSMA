import sys
sys.path.append('../code')

from neuron import h
import cells as cll

import random
import matplotlib.pyplot as plt
import numpy as np


h.load_file('stdrun.hoc')

# Create a list to hold the supraspinal neurons
supraspinal_neurons = []

# Define the parameters
num_supraspinal = 0  # Number of supraspinal neurons
rate_supraspinal = 20  # Firing rate (in Hz)
simulation_duration = 2000  # Simulation duration (in ms)


scs_neurons=[]
num_scs=1
rate_scs= 10

synaptic_weight=0.00037 #this number is set to an EPSP is 212 muV

num_MN=100


record_membrane=[0] #record membrane of these motoneurons


#### Create a population of supraspinal neurons and record its spikes
supraspinal_neurons=[]
for i in range(num_supraspinal):
    syn = h.NetStim()
    syn.interval = 1000.0 / rate_supraspinal  # Inter-spike interval in ms
    syn.number=1e999
    syn.noise=1
    supraspinal_neurons.append(syn)

spike_times2 = [ h.Vector() for i in range(num_supraspinal) ]
spike_detector2 =  [ h.NetCon(supraspinal_neurons[i], None) for i in range(num_supraspinal) ]
for i in range(num_supraspinal): spike_detector2[i].record(spike_times2[i])

#### Create a population of scs pulses and record its spikes
scs_neurons=[]
for i in range(num_scs):
    syn = h.NetStim()
    syn.interval = 1000.0 / rate_scs  # Inter-spike interval in ms
    syn.number=simulation_duration/syn.interval

    syn.noise=0
    scs_neurons.append(syn)

pulse_times = [ h.Vector() for i in range(num_scs) ]
pulse_detector =  [ h.NetCon(scs_neurons[i], None) for i in range(num_scs) ]
for i in range(num_scs): pulse_detector[i].record(pulse_times[i])



drug=False
#### Create a population of MNs  and record its spikes

MNs=[]
spikes_MN=[]
MNs=[ cll.MotoneuronNoDendrites(drug) for imn in range(num_MN) ]

MN_spike_times = [h.Vector() for i in range(num_MN)]
MN_spike_detector=[]
for i in range(num_MN):
    sp_detector=h.NetCon(MNs[i].soma(0.5)._ref_v, None, sec=MNs[i].soma)
    sp_detector.threshold=0
    MN_spike_detector.append(sp_detector)
for i in range(num_MN): MN_spike_detector[i].record(MN_spike_times[i])


#### Connect a population of scs pulses,poisson  and MNs


syns_scs_mn=[]
syns_supraspinal_mn=[]

nc_scs_mn=[]
nc_supraspinal_mn=[]

for imn in range(num_MN):
    syns_scs_mn.append([])
    nc_scs_mn.append([])
    syns_supraspinal_mn.append([])
    nc_supraspinal_mn.append([])
    for iscs in range(num_scs):
        syn_=h.ExpSyn(MNs[imn].soma(0.5))
        syn_.tau=2
        nc=h.NetCon(scs_neurons[iscs], syn_)
        nc.weight[0] = synaptic_weight#+synaptic_weight*0.2*np.random.randn()


        nc_scs_mn.append(nc)
        syns_scs_mn[-1].append(syn_)
    for isupra in range(num_supraspinal):
        syn_ = h.ExpSyn(MNs[imn].soma(0.5))
        syn_.tau = 2
        nc = h.NetCon(supraspinal_neurons[isupra], syn_)
        nc.weight[0] = synaptic_weight #* 0.2*np.random.randn()
        nc_supraspinal_mn[-1].append(nc)
        syns_supraspinal_mn[-1].append(syn_)



#record motoneuron membrane

if len(record_membrane)>0:
    membrane=[]
    time_membrane=[]
    for i in record_membrane:
        membrane.append( h.Vector().record(MNs[i].soma(0.5)._ref_v) )
        time_membrane.append( h.Vector().record(h._ref_t) )

# Create a run control for the simulation
h.finitialize()
h.tstop = simulation_duration
h.run()

for i in range(num_supraspinal):spike_times2[i]=np.array(spike_times2[i])
for i in range(num_scs):pulse_times[i]=np.array(pulse_times[i])

plt.figure()

for i in range(num_supraspinal): plt.plot(spike_times2[i],i*np.ones(len(spike_times2[i])),".")
plt.xlim([0,simulation_duration])
plt.figure()

for i in range(num_scs): plt.plot(pulse_times[i],i*np.ones(len(pulse_times[i])),"k.")
plt.xlim([0,simulation_duration])



plt.figure()
plt.plot(time_membrane[0],membrane[0],"k-")
plt.xlim([0,simulation_duration])

plt.show()


# Access the spike times
#spike_times = [float(spike) for spike in spike_times]

# Now you can analyze or visualize the results using the spike_times data
