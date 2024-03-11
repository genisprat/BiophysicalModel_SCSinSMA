
from sys import path

path.insert(1, '/Users/genis/Dropbox/python3_functions/')
path.insert(1, '/Users/genis/Dropbox/SCS-SMA/neuralnetwork/code/')

import matplotlib.pyplot as plt
import numpy as np
import pickle
#import neuron_functions as nf
import argparse
import importlib as impl
import help_plot as hp
import tools_analysis as tl

impl.reload(hp)
impl.reload(tl)


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42






path_fig="/Users/genis/SCS-SMA_Model_Results/Results/simulations/figures/"

fig,axes=plt.subplots(nrows=1, ncols=1, figsize=(4/2.54,4/2.54))
markersize=2
num_supraspinal = 0  # Number of supraspinal neurons
#num_supraspinal=0
rate_supraspinal = 20  # Firing rate (in Hz)

NUM_SCS=[1, 3, 5, 7, 9, 11,13, 15, 17, 19, 21, 23]



num_MN=1000
rate_scs= 0.5


path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/TMS_experiment/"

# MN_type=["SMA","WT"]
# colors=["red","#666666"]
# labels=[100,0]
#
#
# for itype,mn_type in enumerate(MN_type):
#     P_recruited=np.zeros(len(NUM_SCS))
#     for iscs,num_scs in enumerate(NUM_SCS):
#         Name_file_record="TMS_experiment_"+mn_type+str(num_MN)+"_MNs_supraspinal"+str(num_supraspinal)+"fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"
#
#         f = open(path_simulations + Name_file_record, "rb")
#         data=pickle.load(f)
#         f.close()
#         P_recruited[iscs]=data["P_recruited"]
#
#     stde_precruited=np.sqrt(P_recruited*(1-P_recruited)/num_MN)
#
#     axes.errorbar(NUM_SCS,P_recruited,yerr=stde_precruited,fmt="o-",color=colors[itype],label=mn_type,markersize=markersize)
#



colors=["#cbc9e2","#9e9ac8","#756bb1","#54278f"]
colors=["#bcbddc","#9e9ac8","#807dba","#6a51a3","#4a1486"]
#colors=["#c10a0a","#993232","#705b5b"]


lt=["-","--"]
fontsize=8

num_SMA_MN=[70,49,35,21,0]
num_WT_MN=[0,21,35,49,70]
num_MN=100

labels=[0,30,50,70,100]

for iseverity,num_SMA in enumerate(num_SMA_MN):
    P_recruited=np.zeros(len(NUM_SCS))
    for iscs,num_scs in enumerate(NUM_SCS):

        Name_file_record = "TMS_experiment_SMA" + str(num_SMA) + "WT" + str(num_WT_MN[iseverity]) + "_MNs_supraspinal" + str(
            num_supraspinal) + "fr_" + str(rate_supraspinal) + "_SCSfreq" + str(rate_scs) + "_SCS_num" + str(
            num_scs) + ".pickle"

        f = open(path_simulations + Name_file_record, "rb")
        data=pickle.load(f)
        f.close()
        P_recruited[iscs]=data["P_recruited"]

    stde_precruited=np.sqrt(P_recruited*(1-P_recruited)/num_MN)

    axes.errorbar(NUM_SCS,P_recruited,yerr=stde_precruited,fmt="o-",color=colors[iseverity],label=num_SMA_MN[iseverity],markersize=markersize)





#axes.legend(frameon=False)
axes.set_xlabel("TMS amplitude",fontsize=fontsize)

#axes.set_xlabel("Number of recruited afferenets",fontsize=fontsize)

axes.set_ylabel("Prob. MN recruitment",fontsize=fontsize)
hp.xticks(axes,[0,6,12,18,24],fontsize=fontsize)
hp.yticks(axes,[0,0.5,1],fontsize=fontsize)

name_figure="TMS_experiment2"
hp.remove_axis(axes)
fig.tight_layout()
fig.savefig(path_fig+name_figure+".pdf")
fig.savefig(path_fig+name_figure+".png")




#
#
# duration=40
# delay=2
# sin_f=1/15
# spike_times=[10]
# amplitude=2
# tau=7.5
#
# t,y=tl.dumped_sinus_function(amplitude,tau,sin_f,spike_times,delay,duration)
#
# plt.figure()
#
# plt.plot(t,y,"ok-")


duration=50
delay_mean=10
delay_std=delay_mean*0.2
sin_f=1/15
amplitude_mean=1
amplitude_std=amplitude_mean*0.2
tau_mean=7.5
tau_std=7.5*0.2
dt=0.1





DELAYS=delay_mean+ delay_std*np.random.normal(size=num_MN)
AMPLITUDES= amplitude_mean + amplitude_std*np.random.normal(size=num_MN)
TAUS=tau_mean+tau_std*np.random.normal(size=num_MN)

fig_rec_curve,axes_rec_curve=plt.subplots(nrows=1, ncols=1, figsize=(4/2.54,4/2.54))
fig_reflex,axes_reflex=plt.subplots(nrows=1, ncols=1, figsize=(4/2.54,4/2.54))

num_scs_plot=9

for iseverity,num_SMA in enumerate(num_SMA_MN):
    peak_to_peak=np.zeros(len(NUM_SCS))
    for iscs,num_scs in enumerate(NUM_SCS):

        Name_file_record = "TMS_experiment_SMA" + str(num_SMA) + "WT" + str(
            num_WT_MN[iseverity]) + "_MNs_supraspinal" + str(
            num_supraspinal) + "fr_" + str(rate_supraspinal) + "_SCSfreq" + str(rate_scs) + "_SCS_num" + str(
            num_scs) + ".pickle"

        f = open(path_simulations + Name_file_record, "rb")
        data = pickle.load(f)
        f.close()
        MUAP=np.zeros((num_MN,int(duration/dt)))
        pulse_time=data["SCS_pulses"][0][0]


        for imn in range(len(data["MN_spikes"])):
            if len(data["MN_spikes"][imn])>0: #if there is at least one spike
                data["MN_spikes"][imn]=data["MN_spikes"][imn][data["MN_spikes"][imn]>pulse_time] #disca rd possible spikes produced by initial conditions
                data["MN_spikes"][imn]=data["MN_spikes"][imn]-pulse_time # set time 0 to the SCS pulse
                t, MUAP[imn] = tl.dumped_sinus_function(AMPLITUDES[imn], TAUS[imn], sin_f, data["MN_spikes"][imn], DELAYS[imn], duration)
                #sum_MUAP=sum_MUAP+y
        sum_MUAP=np.sum(MUAP, axis=0)
        peak_to_peak[iscs]=np.max(sum_MUAP)-np.min(sum_MUAP)

        if num_scs == num_scs_plot:
            axes_reflex.plot(t, sum_MUAP, "-", color=colors[iseverity])
            print("num_scs", num_scs)

    axes_rec_curve.plot(NUM_SCS,peak_to_peak,"o-",color=colors[iseverity],label=num_SMA_MN[iseverity],markersize=markersize)





#axes.legend(frameon=False)
axes_rec_curve.set_xlabel("TMS amplitude",fontsize=fontsize)

#axes.set_xlabel("Number of recruited afferenets",fontsize=fontsize)

axes_rec_curve.set_ylabel("Peak to peak (a.u.)",fontsize=fontsize)
hp.xticks(axes_rec_curve,[0,6,12,18,24],fontsize=fontsize)
hp.yticks(axes_rec_curve,[0,20,40],fontsize=fontsize)

name_figure="TMS_experiment_peak_to_peak"
hp.remove_axis(axes_rec_curve)
fig_rec_curve.tight_layout()
fig_rec_curve.savefig(path_fig+name_figure+".pdf")
fig_rec_curve.savefig(path_fig+name_figure+".png")


fig_reflex.savefig(path_fig+"TMSreflexes"+".pdf")

plt.show()




#plt.figure()
#plt.plot(time_membrane[0],membrane[0],"k-")
#plt.xlim([0,simulation_duration])
#plt.show()


# Access the spike times
#spike_times = [float(spike) for spike in spike_times]

# Now you can analyze or visualize the results using the spike_times data
