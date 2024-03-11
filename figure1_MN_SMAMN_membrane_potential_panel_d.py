import sys
sys.path.append('../code')
from tools import general_tools  as gt
from tools import seed_handler as sh
from tools import load_data_tools as ldt
import cells as cll
# from cells import Motoneuron
# from cells import AfferentFiber
# from cells import MotoneuronNoDendrites
from neuron import h

from importlib import reload
from neuron.units import ms, mV
import matplotlib.pyplot as plt
import numpy as np
import pickle


reload(cll)
# try:
#     h.nrn_load_dll("/Users/genis/Dropbox/SCS-SMA/neuralnetwork/code/mod_files/x86_64/libnrnmech.dylib")
# except:
#     print("hola")

drug=False
mn=cll.Motoneuron(drug)
T=5000

amplitude=0.4
duration=T
delay=2

record_membrane=True
h.load_file("stdrun.hoc")

# iclamp=h.IClamp(mnNoDendrites.soma(0.5))
#
# iclamp.delay = 2 #ms
# iclamp.dur = 1000.0 #
# iclamp.amp = 0.5 #nA

#h.load_file('stdrun.hoc')

threshold=0
delay=0
weight=1

#amplitude=0.3
### 0.2 is the amplitude of repeating firing
mnNoDendrites=cll.MotoneuronNoDendrites("WT")

#MN gnbar0.3 decrease
# name="WTgnabar03"
# gnabar_decrease=0.3
# mnNoDendrites.soma.gnabar_motoneuron *= gnabar_decrease

name="WT"
amplitude=0.3

iclamp=mnNoDendrites.current_soma(amplitude,duration,delay)

vec = h.Vector()
netcon = h.NetCon(mnNoDendrites.soma(0.5)._ref_v, None,sec=mnNoDendrites.soma)
netcon.threshold=0
netcon.record(vec)

if record_membrane:
    v = h.Vector().record(mnNoDendrites.soma(0.5)._ref_v)             # Membrane potential vector
    t = h.Vector().record(h._ref_t)                                   # Time stamp vector

h.finitialize(-65 * mV)
h.continuerun(T * ms)
frmn=len(vec)/(T/1000) #in seconds


path_results="/Users/genis/SCS-SMA_Model_Results/Results/Single_Neuron/"
data={}
data["time"]=np.array(t)
data["v"]=np.array(v)
data["amplitude"]=amplitude
data["frmn"]=frmn
file=open(path_results+"membrane_potential_current_"+name+".p","wb")
pickle.dump(data,file)



if record_membrane:
    plt.plot(t,v,'k-')

#Effect of SMA


Namplitudes=15
amplitudes=np.linspace(0.1,0.8,Namplitudes)
amplitudes=np.concatenate((np.array([0.09]),amplitudes))


print("holaSMA")



name="SMA"

mnNoDendritesSMA=cll.MotoneuronNoDendrites("SMA")


amplitude=0.3
iclamp=mnNoDendritesSMA.current_soma(amplitude,duration,delay)
vecSMA = h.Vector()
netconSMA = h.NetCon(mnNoDendritesSMA.soma(0.5)._ref_v, None,sec=mnNoDendritesSMA.soma)
netconSMA.threshold=0
netconSMA.record(vecSMA)


if record_membrane:
    v = h.Vector().record(mnNoDendritesSMA.soma(0.5)._ref_v)             # Membrane potential vector
    t = h.Vector().record(h._ref_t)                                   # Time stamp vector

h.finitialize(-65 * mV)
h.continuerun(T * ms)
frmnSMA=len(vecSMA)/(T/1000.0) #in seconds
if record_membrane:
    plt.plot(t,v,'r-')


data={}
data["time"]=np.array(t)
data["v"]=np.array(v)
data["amplitude"]=amplitude
data["frmn"]=frmnSMA
file=open(path_results+"membrane_potential_current_"+name+".p","wb")
pickle.dump(data,file)
file.close()

plt.show()


