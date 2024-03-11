
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sys import path
path.insert(1, '/Users/genis/Dropbox/SCS-CST/')
path.insert(1, '/Users/genis/Dropbox/python3_functions/')

import help_plot as hp

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

drug=False
T=5000

amplitude=0.4
duration=T
delay=2

record_membrane=True
# iclamp=h.IClamp(mnNoDendrites.soma(0.5))
# iclamp.delay = 2 #ms
# iclamp.dur = 1000.0 #
# iclamp.amp = 0.5 #nA

#h.load_file('stdrun.hoc')



path_results="/Users/genis/SCS-SMA_Model_Results/Results/Single_Neuron/"

name="WT"
file=open(path_results+"membrane_potential_current_"+name+".p","rb")


#file=open(path_results+"membrane_potential_current_MN.p","rb")
data=pickle.load(file)
file.close()
fig,axes=plt.subplots(nrows=1, ncols=1, figsize=(5/2.54,4/2.54))
fig_spike,axes_spike=plt.subplots(nrows=1, ncols=1, figsize=(5/2.54,4/2.54))
fontsize=8

t=np.array(data["time"])
v=np.array(data["v"])
# index=np.where( (t>232) & (t<242))
# axes.plot(t[index]-232,v[index],'k-')

axes.plot(t,v,'k-')

tspike=1947.6
index=np.where( (t>tspike-3) & (t<tspike+10) )
axes_spike.plot(t[index]-tspike,v[index],'k-')


#Effect of SMA
name="SMA"
file=open(path_results+"membrane_potential_current_"+name+".p","rb")

#file=open(path_results+"membrane_potential_current_SMAMN.p","rb") wihtout changing the Sodium
data=pickle.load(file)
file.close()
t=np.array(data["time"])
v=np.array(data["v"])



axes.plot(t,v,'r-')

tspike=1988.8
index=np.where( (t>tspike-3) & (t<tspike+10) )
t_plot=t[index]-tspike
axes_spike.plot(t_plot,v[index],'-',color="#4a1486")

#axes_spike.set_ylabel("Membrane  potential (mV)",fontsize=fontsize)
#axes_spike.set_xlabel("Time (ms)",fontsize=fontsize)

hp.xticks(axes_spike,[0,5,10],fontsize=fontsize)
hp.yticks(axes_spike,[-80,-60],fontsize=fontsize)
axes_spike.set_xlim([t_plot[0],5])

hp.remove_axis(axes_spike)
fig_spike.tight_layout()



axes.set_xlabel("Time (ms)")
axes.set_ylabel("Membrane potential")
fig.tight_layout()
plt.show()





fig_spike.savefig(path_results+"membrane_potential.png")
fig_spike.savefig(path_results+"membrane_potential.pdf")
plt.show()