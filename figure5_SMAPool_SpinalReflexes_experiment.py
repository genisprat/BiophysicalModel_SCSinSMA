import sys
sys.path.append('../code')

from neuron import h
import cells as cll

import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import neuron_functions as nf
import argparse
import importlib as impl

impl.reload(nf)


h.load_file('stdrun.hoc')


record_membrane=[0,1,2] #record membrane of these motoneurons

### MNfr vs supraspinal####

#rate_supraspinal=float(sys.argv[1])

### MNfr vs SCS####
num_scs=int(sys.argv[1])


#def create_exponential_synapses_gamma(source,target,synaptic_weight,shape,tau):




#path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"
path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/SpinalReflexes_experiment/"

# Create a list to hold the supraspinal neurons
supraspinal_neurons = []

# Define the parameters
num_supraspinal = 0  # Number of supraspinal neurons
#num_supraspinal=0
rate_supraspinal = 20  # Firing rate (in Hz)
simulation_duration = 2000  # Simulation duration (in ms)


scs_neurons=[]
#num_scs=14
rate_scs= 0.5
max_num_scs=60
synaptic_weight=0.00037 #this number is set to an EPSP is 212 muV



####SMA sensory mid recovery ########
num_SMA_MN=70
num_WT_MN=0

# num_WT_sensory=35
# num_SMA_sensory=35

num_MN=num_SMA_MN+num_WT_MN
p_recover_sensory=0.0

####SMA sensory and MN mid recovery 1 ########
# num_SMA_MN=35
# num_WT_MN=35
#
# # num_WT_sensory=35
# # num_SMA_sensory=35
#
# num_MN=num_SMA_MN+num_WT_MN
# p_recover_sensory=0.5
#
#
# ####SMA sensory and MN mid recovery 2 ########
#
num_SMA_MN=21
num_WT_MN=49

# num_WT_sensory=35
# num_SMA_sensory=35

num_MN=num_SMA_MN+num_WT_MN
p_recover_sensory=0.25



#p_recovery


SMA_synaptic_weight=synaptic_weight/3 #For SMA sensory


#Name_file_record="SpinalReflexes_experiment_"+MN_type+str(num_MN)+"_MNs_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"

#SMA neuron with WT sensory
#Name_file_record="SpinalReflexes_experiment_"+MN_type+str(num_MN)+"_MNs_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"

#WT MN with SMA sensory
Name_file_record="SpinalReflexes_experimentSMAPool_SMAMN"+str(num_SMA_MN)+"WTMN"+str(num_WT_MN)+"RecoverSensory"+str(int(100*p_recover_sensory))+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"


#Name_file_record="SpinalReflexes_experimentSMAPool_SMAMN"+str(num_SMA_MN)+"SMA_sensory"+str(num_SMA_sensory)+"WTMN"+str(num_WT_MN)+"WTsensory"+str(num_WT_sensory)+"_SCS"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"
#/Users/genis/SCS-SMA_Model_Results/Results/simulations/SpinalReflexes_experiment/SpinalReflexes_experimentSMAPool_SMAMN0SMA_sensory0WTMN70WTSensory70_SCSfreq0.5_SCS_num29.pickle
#SMA MN with SMA sensory
#Name_file_record="SpinalReflexes_experiment_SMAMN-SMAsensory"+str(num_MN)+"_MNs_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"


first_pulse=1500



#### Create a population of scs pulses and record its spikes


scs_neurons=nf.create_input_neurons(num_scs,rate_scs,0,first_spike=first_pulse)
pulse_times=nf.create_spike_recorder_input_neurons(scs_neurons)

#### Create a population of MNs  and record its spikes

MNs=[]
spikes_MN=[]
MNs=[ cll.MotoneuronNoDendrites("WT") for imn in range(num_WT_MN) ]
for imnSMA in range(num_SMA_MN):
    MNs.append(cll.MotoneuronNoDendrites("SMA"))
MN_spike_times=nf.create_spike_recorder_MNs(MNs)



#### Connect a population of scs pulses and poisson  to MNs
shape=1.2
tau=2



np.random.seed(376519)
print(synaptic_weight,SMA_synaptic_weight)

# W_scs_wt=np.random.gamma(shape, scale=synaptic_weight / shape,size=[num_scs,num_WT_MN]) # I put this here so it is always the same
# W_scs_sma=np.zeros((num_scs,num_SMA_MN))
# for i in range(num_scs):
#     for j in range(num_SMA_MN):
#         if np.random.rand()<p_recover_sensory:
#             W_scs_sma[i][j]=np.random.gamma(shape, scale=synaptic_weight / shape)
#         else:
#             W_scs_sma[i][j] = np.random.gamma(shape, scale=SMA_synaptic_weight / shape)

# W_scs_sma=np.random.gamma(shape, scale=SMA_synaptic_weight / shape,size=[num_scs,num_SMA_sensory]) # I put this here so it is always the same
# W_scs_wt=np.random.gamma(shape, scale=synaptic_weight / shape,size=[num_scs,num_WT_sensory]) # I put this here so it is always the same
#W=np.concatenate((W_scs_wt,W_scs_sma),axis=1)

W=np.zeros((num_scs,num_WT_MN+num_SMA_MN))
for i in range(num_scs):
    for j in range(num_WT_MN+num_SMA_MN):
        if np.random.rand()<p_recover_sensory:
            W[i][j]=np.random.gamma(shape, scale=synaptic_weight / shape)
        else:
            W[i][j] = np.random.gamma(shape, scale=SMA_synaptic_weight / shape)



#delays=10+2*np.random.randn(len(W),len(W[0]))
print(np.shape(W),num_scs,np.shape(W))
#W_scs=10*synaptic_weight*np.ones((num_scs,num_MN))

syn_scs,nc_scs=nf.create_exponential_synapses(scs_neurons,MNs,W,tau)

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

plt.figure()
for i,j in enumerate(record_membrane):
    plt.plot(time_membrane[i],membrane[i],"k-")


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
for i in range(num_scs): plt.plot(pulse_times[i],i*np.ones(len(pulse_times[i])),"k.")
plt.xlim([0,simulation_duration])

plt.figure()
for i in range(num_MN): plt.plot(MN_spike_times[i],i*np.ones(len(MN_spike_times[i])),"k.")
plt.xlim([0,simulation_duration])


recruited=0
for ineuron in range(len(MN_spike_times)):
    for ispike in range(len(MN_spike_times[ineuron])):
     if MN_spike_times[ineuron][ispike]>first_pulse:
        recruited+=1

P_recruited=recruited/num_MN

data={}
data["MN_spikes"]=MN_spike_times
data["SCS_pulses"]=pulse_times
data["scs_frequency"]=rate_scs
data["num_scs"]=num_scs
data["supraspinal_rate"]=rate_supraspinal
data["num_supraspinal"]=num_supraspinal
data["simulation_duration"]=simulation_duration
data["num_MN"]=num_MN
data["P_recruited"]=P_recruited
data["synaptic_weight"]=synaptic_weight
print(path_simulations+Name_file_record)
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
