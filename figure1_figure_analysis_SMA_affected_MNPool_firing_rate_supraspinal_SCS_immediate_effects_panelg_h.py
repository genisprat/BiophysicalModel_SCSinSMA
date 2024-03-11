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

num_scs=0
rate_scs= 40
num_MN=10

path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"


NAME_FIG=["MNfr_supraspinal", "MNfr_supraspinal_withSCS"]
#Name_fig="MNfr_SCS"

path_fig="/Users/genis/SCS-SMA_Model_Results/Results/simulations/figures/"

fig,axes=plt.subplots(nrows=1, ncols=1, figsize=(4.2/2.54,3.75/2.54))



#colors=["red","black","blue"]
#colors=["#cbc9e2","#9e9ac8","#756bb1","#54278f"]

colors=["black","#4094E6"]

#colors=[]
lt=["-","--"]
fontsize=7
markersize=2
lw=1
lw_axis=0.5




num_WT_MN=35
num_SMA_MN=35
num_supraspinal=200
labels=["stim OFF","stim ON"]


num_SCS=[0,11]

name_figure="MN_pool_firing_rate_immediate_effect"


for icondition in range(len(num_SCS)):

    num_scs=num_SCS[icondition]
    if num_scs == 0:
        Name_fig = "SMA_affected_MNPool_SMA" + str(num_SMA_MN) + "_WT" + str(num_WT_MN) + "Supraspinal_input" + str(
            num_supraspinal)
    else:
        Name_fig = "SMA_affected_MNPool_SMA" + str(num_SMA_MN) + "_WT" + str(num_WT_MN) + "Supraspinal_input" + str(
            num_supraspinal) + "_SCS" + str(num_scs)

    f=open(path_simulations+Name_fig+".pickle","rb")
    data=pickle.load(f)
    f.close()

    fr_MN=data["fr_MN"]
    parameter=data["rate_supraspinal"] #figure supraspinal
    #parameter=data["num_scs"] #figure num_scs
    fr_mean=np.zeros(len(parameter))
    fr_stde=np.zeros(len(parameter))




    for iparam in range(len(parameter)):
        print(iparam)
        d=( np.array(fr_MN[iparam]),)
        #res = bootstrap(np.array(fr_MN[isupra_spinal]), np.mean, confidence_level=0.95)
        res = bootstrap(d, np.mean, confidence_level=0.95)

        fr_mean[iparam]=np.mean(d)
        fr_stde[iparam]=res.standard_error

    #axes.errorbar(parameter,fr_mean,yerr=fr_stde,fmt="o-",color=colors[itype],label=mn_type) #only stim off
    axes.errorbar(parameter,fr_mean,yerr=fr_stde,fmt="o-",color=colors[icondition],markersize=markersize,label=labels[icondition],lw=lw)

#axes.legend(frameon=False)
axes.set_xlabel("Supraspinal firing rate (Hz)",fontsize=fontsize)

#axes.set_xlabel("Number of recruited afferenets",fontsize=fontsize)

axes.set_ylabel("MN Pool firing rate (Hz)",fontsize=fontsize)

hp.xticks(axes,[0,30,60],fontsize=fontsize)
hp.yticks(axes,[0,700,1400],["0","7","14"],fontsize=fontsize)
axes.set_ylim([-100/2500*1400,1400])
axes.set_xlim([-2,62])
#axes.set_yticks([0,700,1400])
#axes.ticklabel_format(axis="y",style="sci")


for axis in ['top','bottom','left','right']:
    axes.spines[axis].set_linewidth(lw_axis)
axes.tick_params(width=lw_axis)

hp.remove_axis(axes)
fig.tight_layout()
fig.savefig(path_fig+name_figure+".pdf")
fig.savefig(path_fig+name_figure+".png")

plt.show()
#spike_bin=[spike_times_2_spikes_window(data["MN_spikes"][ineuron],window,simulation_duration) for ineuron in range(num_MN) ]