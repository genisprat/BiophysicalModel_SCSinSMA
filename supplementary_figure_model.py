import pickle
from sys import path

import matplotlib.pyplot as plt

path.insert(1, '/Users/genis/Dropbox/python3_functions/')
path.insert(1, '/Users/genis/Dropbox/SCS-SMA/neuralnetwork/code/')
path.append('../code')

import tools_analysis as tl
import numpy as np
import importlib as impl
from scipy.stats import bootstrap
import numpy as np
import help_plot as hp

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ytick.major.pad']='2'
plt.rcParams['xtick.major.pad']='2'
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"


impl.reload(tl)


path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"

path_fig="/Users/genis/SCS-SMA_Model_Results/Results/simulations/figures/"

# Define the parameters
num_supraspinal = 200  # Number of supraspinal neurons

#num_supraspinal=0

rate_supraspinal = 60  # Firing rate (in Hz)
simulation_duration = 4000  # Simulation duration (in ms)

num_scs=0
rate_scs= 40


NUM_SMA_MN=[0,70,49,35,21,0]
NUM_WT_MN=[100,0,21,35,49,70]
#MN_type=sys.argv[2]

window_spike=1


max_force=8595 #Max force value for WT MN Pool



# num_SMA_MN=0
# num_WT_MN=100

#colors=["black","#bcbddc","#9e9ac8","#807dba","#6a51a3","red"]

fig,axes=plt.subplots(nrows=4, ncols=1, figsize=(7/2.54,8/2.54))
name_fig="figure_supplementary_model"

fontsize=7
lw=1
lw_axis=0.5


Force = []
time_force = []
num_SMA_MN=35
num_WT_MN=35
num_supraspinal=200
rate_supraspinal=30

rate_scs=40

NUM_SCS=[0,11]
colors=["black", "#4094E6"]
mean_force=np.zeros(2)
num_scs=11


Name_file_record="figure_supplementary_model_MVC_isometric_task_SMA_MNPool_SMA"+str(num_SMA_MN)+"WT"+str(num_WT_MN)+"MNs_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"

f=open(path_simulations+Name_file_record,"rb")
data=pickle.load(f)
f.close()

MNPool_spikes=np.zeros(int(simulation_duration/window_spike))
MN_spikes=np.zeros((len(data["MN_spikes"]),int(simulation_duration/window_spike)))

for ineuron in range(len(data["MN_spikes"])):
    MN_spikes[ineuron],t_spikes_bin=tl.spike_times_2_spikes_window(data["MN_spikes"][ineuron], window_spike, simulation_duration)
    MNPool_spikes=MNPool_spikes+MN_spikes[ineuron]


F=tl.firing_rate_to_force(MN_spikes,max_force)
Force.append(F)
time_force.append(t_spikes_bin)
#if icondition==0 or icondition==1 or icondition==5:

iplot=0
axes[iplot].plot(time_force[0],F,"k-",lw=lw)
axes[iplot].set_ylabel("Force (a.u.)",fontsize=fontsize)
hp.yticks(axes[iplot],[0,0.25,0.5],fontsize=fontsize)


iplot=1
w=100 #this is ms
s=10
fr,times=tl.firing_rate(MNPool_spikes,w,s)
fr=1000*np.array(fr) #Transform spikes per ms to Hz
axes[iplot].plot(times,fr,"k-",lw=lw)

axes[iplot].set_ylabel("Motoneuron pool \n firing rate (Hz)",fontsize=fontsize)
hp.yticks(axes[iplot],[0,900,1800],[0,9,18],fontsize=fontsize)




iplot=3
for i in range(len(data["SCS_pulses"])):
    axes[iplot].plot(data["SCS_pulses"][i],i*np.ones(len(data["SCS_pulses"][i])),"k|",markersize=0.5)

axes[iplot].set_ylabel("Ia-afferents \n spikes",fontsize=fontsize)
hp.yticks(axes[iplot],[0,5,10],fontsize=fontsize)


iplot=2
for i in range(11):
    axes[iplot].plot(data["supraspinal_spikes"][i],i*np.ones(len(data["supraspinal_spikes"][i])),"k|",markersize=0.5)

axes[iplot].set_ylabel("Supraspinal \n spikes",fontsize=fontsize)
hp.yticks(axes[iplot],[0,5,10],fontsize=fontsize)

for ax in axes:
    ax.set_xlim([1300,3800])
    hp.xticks(ax,[1500,2500,3500],["","",""])
    hp.remove_axis(ax)

hp.xticks(axes[-1],[1500,2500,3500],["0","1","2"],fontsize=fontsize)
axes[-1].set_xlabel("Time (s)",fontsize=fontsize)
for ax in axes:
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw_axis)
    ax.tick_params(width=lw_axis)




fig.tight_layout()

fig.savefig(path_fig+name_fig+".pdf")
plt.show()
#axes.set_xlabel("Number of recruited afferenets",fontsize=fontsize)

#axes.set_ylabel("MN Pool firing rate (Hz)",fontsize=fontsize)
# hp.yticks(axes,[0,0.25,0.5],fontsize=fontsize)
# hp.xticks(axes,[1500,2500,3500], ["0","1","2"],fontsize=fontsize)
# axes.set_xlim([1000,4000]) #I discard the first second to arrive to the stable state
# axes.set_ylim([-0.05,.5])
#
# #hp.yticks(axes,[0,1250,2500],fontsize=fontsize)
#
# for axis in ['top','bottom','left','right']:
#     axes.spines[axis].set_linewidth(lw_axis)
# axes.tick_params(width=lw_axis)
#
# hp.remove_axis(axes)
# fig.tight_layout()
# fig.savefig(path_fig+name_fig+"supraspinal"+str(rate_supraspinal)+".pdf")
# fig.savefig(path_fig+name_fig+"supraspinal"+str(rate_supraspinal)+".png")
#
# fig2.tight_layout()
# fig2.savefig(path_fig+name_fig2+"supraspinal"+str(rate_supraspinal)+".pdf")
# fig2.savefig(path_fig+name_fig2+"supraspinal"+str(rate_supraspinal)+".png")
#
# plt.show()
#



#
# #### Create a population of supraspinal neurons and record its spikes
# MVC_starts=500
# MVC_ends=2500
# supraspinal_neurons=nf.create_input_neurons(num_supraspinal,rate_supraspinal,1,first_spike=MVC_starts)
# supraspinal_spikes=nf.create_spike_recorder_input_neurons(supraspinal_neurons)
#
# #event_stop=[h.NetCon( None,supraspinal_neurons[ineuron]) for ineuron in range(len(supraspinal_neurons))]
#
# event_stop=[]
# for ineuron in range(len(supraspinal_neurons)):
#     aux=h.NetCon( None,supraspinal_neurons[ineuron])
#     aux.weight[0]=-1
#     aux.event(MVC_ends)
#     event_stop.append(aux)
#
# a=h.NetCon( None,supraspinal_neurons[0])
# a.weight[0]=-1
#
#
# def stim_stop():
#     for ineuron in range(len(supraspinal_neurons)):
#         event_stop[ineuron].event(MVC_ends)
#
#
#
#
#
#
# # #### Create a population of scs pulses and record its spikes
# # scs_neurons=nf.create_input_neurons(num_scs,rate_scs,0)
# # pulse_times=nf.create_spike_recorder_input_neurons(scs_neurons)
# #
# # #### Create a population of MNs  and record its spikes
# #
# MNs=[]
# MNs=[ cll.MotoneuronNoDendrites("WT") for imn in range(num_WT_MN) ]
#
# for i in range(num_SMA_MN):
#     MNs.append(cll.MotoneuronNoDendrites("SMA"))
#
# MN_spike_times=nf.create_spike_recorder_MNs(MNs)
#
#
#
# # #### Connect a population of scs pulses and poisson  to MNs
# synaptic_weight=0.00037 #this number is set to an EPSP is 212 muV
# # synaptic_weight_sensory_SMA=0.00037/3 #this number is set to an EPSP is 212 muV
# shape=1.2
# tau=2
# if num_supraspinal>0:
#     W_supraspinal=np.random.gamma(shape, scale=synaptic_weight / shape,size=[num_supraspinal,num_MN])
#     syn_supraspinal,nc_supraspinal=nf.create_exponential_synapses(supraspinal_neurons,MNs,W_supraspinal,tau)
# else:
#     W_supraspinal=0
# # if num_scs>0:
# #     if num_WT_MN>0:
# #         W_scs_WT=np.random.gamma(shape, scale=synaptic_weight / shape,size=[num_scs,num_WT_MN])
# #         syn_scs_WT,nc_scs_WT=nf.create_exponential_synapses(scs_neurons,MNs[:num_WT_MN],W_scs_WT,tau)
# #     else:
# #         W_scs_WT=0
# #
# #     if num_SMA_MN>0:
# #         W_scs_SMA=np.random.gamma(shape, scale=synaptic_weight_sensory_SMA / shape,size=[num_scs,num_SMA_MN])
# #         syn_scs_SMA,nc_scs_SMA=nf.create_exponential_synapses(scs_neurons,MNs[num_SMA_MN:],W_scs_SMA,tau)
# #     else:
# #         W_scs_SMA=0
# # else:
# #     W_scs_WT = 0
# #     W_scs_SMA = 0
# #
# # #record motoneuron membrane
# #
# # if len(record_membrane)>0:
# #     membrane=[]
# #     time_membrane=[]
# #     for i in record_membrane:
# #         membrane.append( h.Vector().record(MNs[i].soma(0.5)._ref_v) )
# #         time_membrane.append( h.Vector().record(h._ref_t) )
# #
# # # Create a run control for the simulation
# #
#
# fih=h.FInitializeHandler(2,stim_stop)
# h.finitialize()
# h.tstop = simulation_duration
# h.run()
#
#
#
#
# # h.FInitializeHandler("event_stop[0].event(800)")
# # h.finitialize()
# # h.tstop = simulation_duration
# # h.run()
# #
# for i in range(num_supraspinal):
#     if len(supraspinal_spikes[i])>0:
#         supraspinal_spikes[i]=np.array(supraspinal_spikes[i])
#     else:
#         supraspinal_spikes[i] = np.array([])
# # for i in range(num_scs):
# #     if len(pulse_times[i])>0:
# #         pulse_times[i]=np.array(pulse_times[i])
# #     else:
# #         pulse_times[i] = np.array([])
#
# for i in range(num_MN):
#     if len(MN_spike_times[i])>0:
#         MN_spike_times[i]=np.array(MN_spike_times[i])
#     else:
#         MN_spike_times[i]=np.array([])
#
# # plt.figure()
# #
# # for i in range(num_supraspinal): plt.plot(supraspinal_spikes[i],i*np.ones(len(supraspinal_spikes[i])),".")
# # plt.xlim([0,simulation_duration])
# # plt.figure()
#
# # for i in range(num_scs): plt.plot(pulse_times[i],i*np.ones(len(pulse_times[i])),"k.")
# # plt.xlim([0,simulation_duration])
#
# # plt.figure()
# # for i in range(num_MN): plt.plot(MN_spike_times[i],i*np.ones(len(MN_spike_times[i])),"k.")
# # plt.xlim([0,simulation_duration])
#
# data={}
# data["MN_spikes"]=MN_spike_times
# #data["SCS_pulses"]=pulse_times
# data["supraspinal_spikes"]=supraspinal_spikes
# data["scs_frequency"]=rate_scs
# data["num_scs"]=num_scs
# data["supraspinal_rate"]=rate_supraspinal
# data["num_supraspinal"]=num_supraspinal
# data["simulation_duration"]=simulation_duration
# data["num_MN"]=num_MN
# data["num_SMA_MN"]=num_SMA_MN
# data["num_WT_MN"]=num_WT_MN
# data["synaptic_weight"]=synaptic_weight
# #data["synaptic_weight_sensory_SMA"]=synaptic_weight_sensory_SMA
# data["W_supraspinal"]=W_supraspinal
# #data["W_scs_SMA"]=W_scs_SMA
# #data["W_scs_WT"]=W_scs_WT
#
#
#
#
#
# f=open(path_simulations+Name_file_record,"wb")
# pickle.dump(data,f)
# f.close()
#
#
#
# #plt.figure()
# #plt.plot(time_membrane[0],membrane[0],"k-")
# #plt.xlim([0,simulation_duration])
# #plt.show()
#
#
# # Access the spike times
# #spike_times = [float(spike) for spike in spike_times]
#
# # Now you can analyze or visualize the results using the spike_times data