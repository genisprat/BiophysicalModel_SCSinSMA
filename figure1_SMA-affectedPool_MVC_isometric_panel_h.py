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


record_membrane=[0,1] #record membrane of these motoneurons

### MNfr vs supraspinal####

#rate_supraspinal=float(sys.argv[1])

### MNfr vs SCS####
#num_scs=int(sys.argv[1])


#def create_exponential_synapses_gamma(source,target,synaptic_weight,shape,tau):




#path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"
path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"

# Create a list to hold the supraspinal neurons
supraspinal_neurons = []

# Define the parameters
num_supraspinal = 200  # Number of supraspinal neurons

#num_supraspinal=0

rate_supraspinal = 20  # Firing rate (in Hz)
simulation_duration = 4000  # Simulation duration (in ms)


scs_neurons=[]
num_scs=11
rate_scs= 40

#synaptic_weight=0.00037 #this number is set to an EPSP is 212 muV

# WT MN pool
# num_SMA_MN=0
# num_WT_MN=100

# # 100% SMA, 70 survival
# num_SMA_MN=70
# num_WT_MN=0
#
# # 70% SMA, 70 survival
# num_SMA_MN=49
# num_WT_MN=21
#
# # 50% SMA, 70 survival
# num_SMA_MN=35
# num_WT_MN=35
# #
# # # 30% SMA, 70 survival
# num_SMA_MN=21
# num_WT_MN=49
# #
# # # # 0% SMA, 70 survival
# num_SMA_MN=0
# num_WT_MN=70


percentatge_SMA=70
num_MN=30
num_SMA_MN=int(num_MN*percentatge_SMA/100)
num_WT_MN=num_MN-num_SMA_MN
num_MN=num_SMA_MN+num_WT_MN


#MN_type=sys.argv[2]

#Name_file_record=MN_type+str(num_MN)+"_prova_MNs_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"
Name_file_record="MVC_isometric_task_SMA_MNPool_SMA"+str(num_SMA_MN)+"WT"+str(num_WT_MN)+"MNs_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"


#### Create a population of supraspinal neurons and record its spikes
MVC_starts=1500
MVC_ends=3500
supraspinal_neurons=nf.create_input_neurons(num_supraspinal,rate_supraspinal,1,first_spike=MVC_starts)
supraspinal_spikes=nf.create_spike_recorder_input_neurons(supraspinal_neurons)

#event_stop=[h.NetCon( None,supraspinal_neurons[ineuron]) for ineuron in range(len(supraspinal_neurons))]

event_stop=[]
for ineuron in range(len(supraspinal_neurons)):
    aux=h.NetCon( None,supraspinal_neurons[ineuron])
    aux.weight[0]=-1
    aux.event(MVC_ends)
    event_stop.append(aux)

a=h.NetCon( None,supraspinal_neurons[0])
a.weight[0]=-1


def stim_stop():
    for ineuron in range(len(supraspinal_neurons)):
        event_stop[ineuron].event(MVC_ends)






# #### Create a population of scs pulses and record its spikes
# scs_neurons=nf.create_input_neurons(num_scs,rate_scs,0)
# pulse_times=nf.create_spike_recorder_input_neurons(scs_neurons)
#
# #### Create a population of MNs  and record its spikes
#
MNs=[]
MNs=[ cll.MotoneuronNoDendrites("WT") for imn in range(num_WT_MN) ]

for i in range(num_SMA_MN):
    MNs.append(cll.MotoneuronNoDendrites("SMA"))

MN_spike_times=nf.create_spike_recorder_MNs(MNs)



# #### Connect a population of scs pulses and poisson  to MNs
synaptic_weight=0.00037 #this number is set to an EPSP is 212 muV
# synaptic_weight_sensory_SMA=0.00037/3 #this number is set to an EPSP is 212 muV
shape=1.2
tau=2
if num_supraspinal>0:
    W_supraspinal=np.random.gamma(shape, scale=synaptic_weight / shape,size=[num_supraspinal,num_MN])
    syn_supraspinal,nc_supraspinal=nf.create_exponential_synapses(supraspinal_neurons,MNs,W_supraspinal,tau)
else:
    W_supraspinal=0
# if num_scs>0:
#     if num_WT_MN>0:
#         W_scs_WT=np.random.gamma(shape, scale=synaptic_weight / shape,size=[num_scs,num_WT_MN])
#         syn_scs_WT,nc_scs_WT=nf.create_exponential_synapses(scs_neurons,MNs[:num_WT_MN],W_scs_WT,tau)
#     else:
#         W_scs_WT=0
#
#     if num_SMA_MN>0:
#         W_scs_SMA=np.random.gamma(shape, scale=synaptic_weight_sensory_SMA / shape,size=[num_scs,num_SMA_MN])
#         syn_scs_SMA,nc_scs_SMA=nf.create_exponential_synapses(scs_neurons,MNs[num_SMA_MN:],W_scs_SMA,tau)
#     else:
#         W_scs_SMA=0
# else:
#     W_scs_WT = 0
#     W_scs_SMA = 0
#
# #record motoneuron membrane
#
# if len(record_membrane)>0:
#     membrane=[]
#     time_membrane=[]
#     for i in record_membrane:
#         membrane.append( h.Vector().record(MNs[i].soma(0.5)._ref_v) )
#         time_membrane.append( h.Vector().record(h._ref_t) )
#
# # Create a run control for the simulation
#

fih=h.FInitializeHandler(2,stim_stop)
h.finitialize()
h.tstop = simulation_duration
h.run()




# h.FInitializeHandler("event_stop[0].event(800)")
# h.finitialize()
# h.tstop = simulation_duration
# h.run()
#
for i in range(num_supraspinal):
    if len(supraspinal_spikes[i])>0:
        supraspinal_spikes[i]=np.array(supraspinal_spikes[i])
    else:
        supraspinal_spikes[i] = np.array([])
# for i in range(num_scs):
#     if len(pulse_times[i])>0:
#         pulse_times[i]=np.array(pulse_times[i])
#     else:
#         pulse_times[i] = np.array([])

for i in range(num_MN):
    if len(MN_spike_times[i])>0:
        MN_spike_times[i]=np.array(MN_spike_times[i])
    else:
        MN_spike_times[i]=np.array([])

# plt.figure()
#
# for i in range(num_supraspinal): plt.plot(supraspinal_spikes[i],i*np.ones(len(supraspinal_spikes[i])),".")
# plt.xlim([0,simulation_duration])
# plt.figure()

# for i in range(num_scs): plt.plot(pulse_times[i],i*np.ones(len(pulse_times[i])),"k.")
# plt.xlim([0,simulation_duration])

# plt.figure()
# for i in range(num_MN): plt.plot(MN_spike_times[i],i*np.ones(len(MN_spike_times[i])),"k.")
# plt.xlim([0,simulation_duration])

data={}
data["MN_spikes"]=MN_spike_times
#data["SCS_pulses"]=pulse_times
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
#data["synaptic_weight_sensory_SMA"]=synaptic_weight_sensory_SMA
data["W_supraspinal"]=W_supraspinal
#data["W_scs_SMA"]=W_scs_SMA
#data["W_scs_WT"]=W_scs_WT





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