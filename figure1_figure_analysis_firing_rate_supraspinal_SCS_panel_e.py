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
plt.rcParams['ps.fonttype'] = 42


impl.reload(tl)

num_scs=0
rate_scs= 40
num_MN=10

path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"


NAME_FIG=["MNfr_supraspinal", "MNfr_supraspinal_withSCS"]
#Name_fig="MNfr_SCS"

path_fig="/Users/genis/SCS-SMA_Model_Results/Results/simulations/figures/"

fig,axes=plt.subplots(nrows=1, ncols=1, figsize=(10/2.54,9/2.54))


MN_type=["SMA","WT"]
colors=["red","black"]
lt=["-","--"]
fontsize=8

name_figure="MNfr_vs_Supraspinal_long_term_effect"

for iname, Name_fig in enumerate(NAME_FIG):

    for itype,mn_type in enumerate(MN_type):
        f=open(path_simulations+Name_fig+"MN_type"+mn_type+".pickle","rb")
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

            fr_mean[iparam]=np.mean(res.bootstrap_distribution)
            fr_stde[iparam]=res.standard_error

        #axes.errorbar(parameter,fr_mean,yerr=fr_stde,fmt="o-",color=colors[itype],label=mn_type) #only stim off
        axes.errorbar(parameter,fr_mean,yerr=fr_stde,fmt=lt[iname],color=colors[itype],label=mn_type)

axes.legend(frameon=False)
axes.set_xlabel("Supraspinal firing rate (Hz)")

#axes.set_xlabel("Number of recruited afferenets",fontsize=fontsize)

axes.set_ylabel("MNs firing rate (Hz)",fontsize=fontsize)

hp.remove_axis(axes)
fig.tight_layout()
fig.savefig(path_fig+name_figure+".pdf")
fig.savefig(path_fig+name_figure+".png")

plt.show()
#spike_bin=[spike_times_2_spikes_window(data["MN_spikes"][ineuron],window,simulation_duration) for ineuron in range(num_MN) ]