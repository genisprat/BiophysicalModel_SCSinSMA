import pickle
from sys import path
path.insert(1, '/Users/genis/Dropbox/python3_functions/')
path.insert(1, '/Users/genis/Dropbox/SCS-SMA/neuralnetwork/code/')
path.append('../code')

import tools_analysis as tl
import numpy as np
import importlib as impl
from importlib import reload

import matplotlib.pyplot as plt
import help_plot as hp
reload(hp)

impl.reload(tl)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

num_scs=0
NUM_SCS=num_scs
rate_scs= 40
rate_supraspinal=60
num_supraspinal=200


path_fig="/Users/genis/SCS-SMA_Model_Results/Results/simulations/figures/"

name_figure="SMA_affected_MNPool_colormap"
path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"

fr=[]
PERCENTATGE_SMA=[0,10,20,30,40,50,60,70,80,90,100]
PERCENTATGE_SMA_ticks=[0,20,40,60,80,100]

NUM_MN=[10,20,30,40,50,60,70,80,90,100]
NUM_MN_label=100-np.array([10,20,30,40,50,60,70,80,90,100]) #I use this to use motoneuron loss instead of number of MN
NUM_MN_label_ticks=100-np.array([10,40,70,100])


mean_fr=np.zeros((len(NUM_MN),len(PERCENTATGE_SMA)))
for inum_MN, num_MN in enumerate(NUM_MN):

    for ipercent, percentatge_SMA in enumerate(PERCENTATGE_SMA): #analysis supraspinal
        num_SMA_MN = int(num_MN * percentatge_SMA / 100.0)
        num_WT_MN = num_MN - num_SMA_MN

        Name_file_record = "SMA_MNPool_SMA" + str(num_SMA_MN) + "WT" + str(num_WT_MN) + "MNs_supraspinal" + str(
            num_supraspinal) + "fr_" + str(rate_supraspinal) + "_SCSfreq" + str(rate_scs) + "_SCS_num" + str(
            num_scs) + ".pickle"

        path_file=path_simulations+Name_file_record
        f=open(path_file,"rb")
        data=pickle.load(f)
        f.close()

        window=100
        num_MN=len(data["MN_spikes"])
        simulation_duration=data["simulation_duration"]


        ineuron=0
        #spike_times_2_spikes_window(data["MN_spikes"][ineuron],window,simulation_duration)
        spike_bins=[]
        #fr=np.zeros(num_MN)
        #transform from spike_times to spike vs time widows of duration window
        for ineuron in range(num_MN):
            #print(ineuron)
            x,t_spike_bins=tl.spike_times_2_spikes_window(data["MN_spikes"][ineuron], window, simulation_duration)
            spike_bins.append(x)


        spike_bins=np.array(spike_bins)
        spike_bins_all=np.sum(spike_bins, axis=0)
        t0=1000
        tf=simulation_duration
        it0=int(t0/window)
        itf=int(tf/window)
        fr.append(spike_bins_all[it0:itf]*1000/window) # I only add after 1 second where firing rate is more stable, because its firing rate I want this to be in Hz
        #fr.append( [1000*np.sum(spike_bins[ineuron][it0:itf])/(simulation_duration-t0) for ineuron in range(num_MN) ] ) #multiply by 1000 to have Hz

        mean_fr[inum_MN][ipercent]=np.mean(fr[-1])


fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(7/2.54, 7/2.54))

fontsize_ticks=7
fontsize=8


#im=axes.imshow(mean_fr,extent=[NUM_MN[0]-5,NUM_MN[-1]+5,PERCENTATGE_SMA[0]-5,PERCENTATGE_SMA[-1]+5],origin='lower',cmap='inferno',vmax=2400)

im=axes.imshow(mean_fr,extent=[PERCENTATGE_SMA[0]-5,PERCENTATGE_SMA[-1]+5,NUM_MN_label[-1]-5,NUM_MN_label[0]+5],cmap='inferno_r',vmax=2400)


cax = axes.inset_axes([1.04, 0.0, 0.05, 1.0], transform=axes.transAxes)
fig.colorbar(im, ax=axes, cax=cax)
yticks_cbar=[150,900,1650,2400]
cbar=fig.colorbar(im,ax=axes,cax=cax,ticks=yticks_cbar)
cbar.ax.set_yticklabels(["150 Hz","900","1650","2400"],fontsize=fontsize_ticks)



hp.yticks(axes,NUM_MN_label_ticks,fontsize=fontsize_ticks)
hp.xticks(axes,PERCENTATGE_SMA_ticks,fontsize=fontsize_ticks)
axes.set_ylabel("Percentatge of MN loss",fontsize=fontsize)
axes.set_xlabel("Percentatge of SMA-affected MNs",fontsize=fontsize)



fig.tight_layout()
fig.savefig(path_fig+name_figure+".pdf")
fig.savefig(path_fig+name_figure+".png")


#im=axes[0].imshow(firing_rates,extent=[amplitudes[0]-0.05,amplitudes[-1]+0.05,0,len(frequencies)],aspect="auto",origin='lower',cmap='inferno',vmax=vmax,vmin=0.0)

#
# data={}
#
# data["fr_MN"]=fr
# data["rate_supraspinal"]=RATE_SUPRASPINAL
# data["num_SMA_MN"]=num_SMA_MN
# data["num_WT_MN"]=num_WT_MN
# data["window"]=window # Time window in ms
#
#
#
#
#
# #
# f=open(path_simulations+Name_fig+".pickle","wb")
# pickle.dump(data,f)
# f.close()

#spike_bin=[spike_times_2_spikes_window(data["MN_spikes"][ineuron],window,simulation_duration) for ineuron in range(num_MN) ]