import sys
sys.path.append('../code')

from neuron import h
import cells as cll

import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import neuron_functions as nf

import importlib as impl

impl.reload(nf)


h.load_file('stdrun.hoc')

MN_type=sys.argv[2]


record_membrane=[0,1] #record membrane of these motoneurons
num_MN=100


### MNfr vs supraspinal####
rate_supraspinal=float(sys.argv[1])
num_supraspinal=200
num_scs= 0  # Firing rate (in Hz)
rate_scs= 40

Name_file_record="Only_supraspinal"+MN_type+str(num_MN)+"_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"


### MNfr vs SCS####
# num_scs=int(sys.argv[1])
# num_supraspinal=0
# rate_supraspinal = 20  # Firing rate (in Hz)
# rate_scs= 40
#Name_file_record="Only_SCS"+MN_type+str(num_MN)+"_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"

#def create_exponential_synapses_gamma(source,target,synaptic_weight,shape,tau):




#path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"
path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_SCS/"

# Create a list to hold the supraspinal neurons
supraspinal_neurons = []

# Define the parameters
#num_supraspinal = 600  # Number of supraspinal neurons



simulation_duration = 5000  # Simulation duration (in ms)




#synaptic_weight=0.00037 #this number is set to an EPSP is 212 muV





#### Create a population of supraspinal neurons and record its spikes
supraspinal_neurons=nf.create_input_neurons(num_supraspinal,rate_supraspinal,1)
supraspinal_spikes=nf.create_spike_recorder_input_neurons(supraspinal_neurons)

#### Create a population of scs pulses and record its spikes
scs_neurons=nf.create_input_neurons(num_scs,rate_scs,0)
pulse_times=nf.create_spike_recorder_input_neurons(scs_neurons)

#### Create a population of MNs  and record its spikes

MNs=[]
MNs=[ cll.MotoneuronNoDendrites(MN_type) for imn in range(num_MN) ]
MN_spike_times=nf.create_spike_recorder_MNs(MNs)


#### Connect a population of scs pulses and poisson  to MNs
synaptic_weight_WT=0.00037 #this number is set to an EPSP is 212 muV
shape=1.2
tau=2
W_supraspinal=np.random.gamma(shape, scale=synaptic_weight_WT / shape,size=[num_supraspinal,num_MN])
syn_supraspinal,nc_supraspinal=nf.create_exponential_synapses(supraspinal_neurons,MNs,W_supraspinal,tau)

synaptic_weight_sensory_SMA=0.00037/1.5 #this number is set to an EPSP is 212 muV
if MN_type=="SMA":
    W_scs_SMA=np.random.gamma(shape, scale=synaptic_weight_sensory_SMA / shape,size=[num_scs,num_MN])
elif MN_type=="WT":
    W_scs_SMA = np.random.gamma(shape, scale=synaptic_weight_WT / shape, size=[num_scs, num_MN])

syn_scs,nc_scs=nf.create_exponential_synapses(scs_neurons,MNs,W_scs_SMA,tau)


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
plt.figure()

for i in range(num_scs): plt.plot(pulse_times[i],i*np.ones(len(pulse_times[i])),"k.")
plt.xlim([0,simulation_duration])

plt.figure()
for i in range(num_MN): plt.plot(MN_spike_times[i],i*np.ones(len(MN_spike_times[i])),"k.")
plt.xlim([0,simulation_duration])


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
data["synaptic_weight_WT"]=synaptic_weight_WT
data["synaptic_weight_sensory_SMA"]=synaptic_weight_sensory_SMA

f=open(path_simulations+Name_file_record,"wb")
pickle.dump(data,f)
f.close()



#plt.figure()
#plt.plot(time_membrane[0],membrane[0],"k-")
#plt.xlim([0,simulation_duration])
#plt.show()


# Access the spike times
#spike_times = [float(spike) for spike in spike_times]

# Now you can analyze or visualize the results using the spike_times data
