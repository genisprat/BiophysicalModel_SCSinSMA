import sys
sys.path.append('../code')

from neuron import h
import cells as cll
import time as time
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import neuron_functions as nf

import importlib as impl

t0=time.time()

impl.reload(nf)


h.load_file('stdrun.hoc')



### MNfr vs supraspinal####

rate_supraspinal=20


#path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"
path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"

# Create a list to hold the supraspinal neurons
supraspinal_neurons = []

# Define the parameters
num_supraspinal = 200  # Number of supraspinal neurons

#num_supraspinal=0

rate_supraspinal = 20  # Firing rate (in Hz)
simulation_duration = 1000  # Simulation duration (in ms)


### immediate effects figure #####
scs_neurons=[]
num_scs=11
rate_scs= 40

# # 50% SMA, 70 survival
num_SMA_MN=35
num_WT_MN=35
num_MN=num_SMA_MN+num_WT_MN
p_recover_sensory=0.5



Name_file_record="Dema_SMA_MNPool_SMA"+str(num_SMA_MN)+"WT"+str(num_WT_MN)+"MNs_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"


#### Create a population of supraspinal neurons and record its spikes
supraspinal_neurons=nf.create_input_neurons(num_supraspinal,rate_supraspinal,1)
supraspinal_spikes=nf.create_spike_recorder_input_neurons(supraspinal_neurons)

#### Create a population of scs pulses and record its spikes
scs_neurons=nf.create_input_neurons(num_scs,rate_scs,0)
pulse_times=nf.create_spike_recorder_input_neurons(scs_neurons)

#### Create a population of MNs  and record its spikes

MNs=[]
MNs=[ cll.MotoneuronNoDendrites("WT") for imn in range(num_WT_MN) ]

for i in range(num_SMA_MN):
    MNs.append(cll.MotoneuronNoDendrites("SMA"))

MN_spike_times=nf.create_spike_recorder_MNs(MNs)



#### Connect a population of scs pulses and poisson  to MNs
synaptic_weight=0.00037 #this number is set to an EPSP is 212 muV
SMA_synaptic_weight=0.00037/3 #this number is set to an EPSP is 212 muV
shape=1.2
tau=2

W_scs=np.zeros((num_scs,num_WT_MN+num_SMA_MN))
for i in range(num_scs):
    for j in range(num_WT_MN+num_SMA_MN):
        if np.random.rand()<p_recover_sensory:
            W_scs[i][j]=np.random.gamma(shape, scale=synaptic_weight / shape)
        else:
            W_scs[i][j] = np.random.gamma(shape, scale=SMA_synaptic_weight / shape)

syn_scs,nc_scs=nf.create_exponential_synapses(scs_neurons,MNs,W_scs,tau)


if num_supraspinal>0:
    W_supraspinal=np.random.gamma(shape, scale=synaptic_weight / shape,size=[num_supraspinal,num_MN])
    syn_supraspinal,nc_supraspinal=nf.create_exponential_synapses(supraspinal_neurons,MNs,W_supraspinal,tau)
else:
    W_supraspinal=0


# Create a run control for the simulation
h.finitialize()
h.tstop = simulation_duration
h.run()

for i in range(num_supraspinal):
    if len(supraspinal_spikes[i])>0:
        supraspinal_spikes[i]=np.array(supraspinal_spikes[i])
    else:
        supraspinal_spikes[i] = np.array([])
for i in range(num_scs):
    if len(pulse_times[i])>0:
        pulse_times[i]=np.array(pulse_times[i])
    else:
        pulse_times[i] = np.array([])

for i in range(num_MN):
    if len(MN_spike_times[i])>0:
        MN_spike_times[i]=np.array(MN_spike_times[i])
    else:
        MN_spike_times[i]=np.array([])

plt.figure()

for i in range(num_supraspinal): plt.plot(supraspinal_spikes[i],i*np.ones(len(supraspinal_spikes[i])),".")
plt.xlim([0,simulation_duration])
plt.xticks([0,simulation_duration/2,simulation_duration])
plt.title("Supraspinal inputs")
plt.xlabel("Time (ms)")


plt.figure()
for i in range(num_scs): plt.plot(pulse_times[i],i*np.ones(len(pulse_times[i])),"k.")
plt.xticks([0,simulation_duration/2,simulation_duration])
plt.title("Sensory Afferents")
plt.xlim([0,simulation_duration])
plt.xlabel("Time (ms)")


plt.figure()
for i in range(num_MN): plt.plot(MN_spike_times[i],i*np.ones(len(MN_spike_times[i])),"k.")
plt.xlim([0,simulation_duration])
plt.xticks([0,simulation_duration/2,simulation_duration])
plt.title("SMA-affected Motoneuron Pool")
plt.xlabel("Time (ms)")




data={}
data["MN_spikes"]=MN_spike_times
data["SCS_pulses"]=pulse_times
data["supraspinal_spikes"]=supraspinal_spikes
data["scs_frequency"]=rate_scs
data["num_scs"]=num_scs
data["supraspinal_rate"]=rate_supraspinal
data["num_supraspinal"]=num_supraspinal
data["simulation_duration"]=simulation_duration
data["num_MN"]=num_MN
data["num_SMA_MN"]=num_SMA_MN
data["num_WT_MN"]=num_WT_MN
data["synaptic_weight"]=synaptic_weight
data["synaptic_weight_sensory_SMA"]=SMA_synaptic_weight
data["W_supraspinal"]=W_supraspinal
data["W_scs"]=W_scs





f=open(path_simulations+Name_file_record,"wb")
pickle.dump(data,f)
f.close()

t1=time.time()
print(t1-t0)

#plt.figure()
#plt.plot(time_membrane[0],membrane[0],"k-")
#plt.xlim([0,simulation_duration])
#plt.show()


# Access the spike times
#spike_times = [float(spike) for spike in spike_times]

# Now you can analyze or visualize the results using the spike_times data
