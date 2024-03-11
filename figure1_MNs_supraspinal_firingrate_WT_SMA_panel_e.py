from sys import path
path.insert(1, '/Users/genis/Dropbox/python3_functions/')
path.insert(1, '/Users/genis/Dropbox/SCS-SMA/neuralnetwork/code/')
path.append('../code')
from neuron import h
import cells as cll

import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import neuron_functions as nf
import help_plot as hp

import importlib as impl

impl.reload(nf)


h.load_file('stdrun.hoc')

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


record_membrane=[] #record membrane of these motoneurons


h.Random(94729)
np.random.seed(87456)


### MNfr vs supraspinal####

#rate_supraspinal=float(sys.argv[1])

### MNfr vs SCS####


#def create_exponential_synapses_gamma(source,target,synaptic_weight,shape,tau):




#path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"
path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_SCS/"

# Create a list to hold the supraspinal neurons
supraspinal_neurons = []

# Define the parameters
#num_supraspinal = 600  # Number of supraspinal neurons
num_supraspinal=200
rate_supraspinal = 60  # Firing rate (in Hz)


simulation_duration = 1200  # Simulation duration (in ms)


scs_neurons=[]
num_scs=0
rate_scs= 40

#synaptic_weight=0.00037 #this number is set to an EPSP is 212 muV

num_MN=10
record_membrane=range(num_MN) #record membrane of these motoneurons

#MN_type=sys.argv[2]

#Name_file_record="Only_SCS"+MN_type+str(num_MN)+"_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"
Name_file_record="membrane_potential_WT_SMA"+str(num_MN)+"_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"

#### Create a population of supraspinal neurons and record its spikes
supraspinal_neurons=nf.create_input_neurons(num_supraspinal,rate_supraspinal,1)
supraspinal_spikes=nf.create_spike_recorder_input_neurons(supraspinal_neurons)

#### Create a population of scs pulses and record its spikes
scs_neurons=nf.create_input_neurons(num_scs,rate_scs,0)
pulse_times=nf.create_spike_recorder_input_neurons(scs_neurons)

#### Create a population of 1 WT MNs and 1 SMA  and record its spikes
MNs=[]
for iMN in range(int(num_MN/2)):MNs.append((cll.MotoneuronNoDendrites("WT")))
for iMN in range(int(num_MN/2)):MNs.append((cll.MotoneuronNoDendrites("SMA")))

#MNs=[cll.MotoneuronNoDendrites("WT"),cll.MotoneuronNoDendrites("SMA")]
MN_spike_times=nf.create_spike_recorder_MNs(MNs)


#### Connect a population of scs pulses and poisson  to MNs
synaptic_weight_WT=0.00037 #this number is set to an EPSP is 212 muV
shape=1.2
tau=2
W_supraspinal=np.random.gamma(shape, scale=synaptic_weight_WT / shape,size=[num_supraspinal,num_MN]) #same connections for WT and SMA

#W_supraspinal2=np.zeros((num_supraspinal,num_MN))
# W_supraspinal2[:,0]=W_supraspinal[:,0]
# W_supraspinal2[:,1]=W_supraspinal[:,0]
syn_supraspinal,nc_supraspinal=nf.create_exponential_synapses(supraspinal_neurons,MNs,W_supraspinal,tau)

synaptic_weight_sensory_SMA=0.00037/1.5 #this number is set to an EPSP is 212 muV


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


for i in range(num_MN):
    if len(MN_spike_times[i])>0:
        MN_spike_times[i]=np.array(MN_spike_times[i])
    else:
        MN_spike_times[i]=np.array([])

plt.figure()

for i in range(num_supraspinal): plt.plot(supraspinal_spikes[i],i*np.ones(len(supraspinal_spikes[i])),".")
plt.xlim([0,simulation_duration])



plt.figure()
for i in range(num_MN): plt.plot(MN_spike_times[i],i*np.ones(len(MN_spike_times[i])),"k.")
plt.xlim([0,simulation_duration])

fig,axes=plt.subplots(nrows=1, ncols=1, figsize=(11/2.54,7/2.54))
fontsize=12
ineuron_WT=0
ineuronSMA=5
axes.plot(time_membrane[ineuron_WT],membrane[ineuron_WT],"k-")
axes.plot(time_membrane[ineuronSMA],membrane[ineuronSMA],"r-")

axes.set_xlabel("Time (ms)",fontsize=fontsize)

#axes.set_xlabel("Number of recruited afferenets",fontsize=fontsize)

axes.set_ylabel("Membrane potential (mV)",fontsize=fontsize)
#hp.xticks(axes,[200,600,1000],["0","400","800"],fontsize=fontsize)
hp.yticks(axes,[-80,-60,-40,-20,0,20],fontsize=fontsize)
axes.set_xlim([75,320])

hp.remove_axis(axes)
fig.tight_layout()
path_fig="/Users/genis/SCS-SMA_Model_Results/Results/simulations/figures/"
name_figure="membrane_potential_firing_rate_WT_vs_SMA"
fig.savefig(path_fig+name_figure+".pdf")
fig.savefig(path_fig+name_figure+".png")

plt.show()



#plt.figure()
#plt.plot(time_membrane[0],membrane[0],"k-")
#plt.xlim([0,simulation_duration])
#plt.show()


# Access the spike times
#spike_times = [float(spike) for spike in spike_times]

# Now you can analyze or visualize the results using the spike_times data
