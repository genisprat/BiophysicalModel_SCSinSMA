import pickle
from sys import path
path.insert(1, '/Users/genis/Dropbox/python3_functions/')
path.insert(1, '/Users/genis/Dropbox/SCS-SMA/neuralnetwork/code/')
path.append('../code')

import tools_analysis as tl
import numpy as np
import importlib as impl

impl.reload(tl)

MN_type="SMA"


# RATE_SUPRASPINAL=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85, 90, 95, 100, 200]
# RATE_SUPRASPINAL_str=["5.0","10.0","15.0","20.0","25.0","30.0","35.0","40.0","45.0","50.0","55.0","60.0","65.0","70.0","75.0","80.0","85.0","90.0"
#                       ,"95.0","100.0","200.0"]

### analysis of rate MNs vs rate supraspinal
RATE_SUPRASPINAL=[1,5,10,15,20,25,30,35,40,45,50,55,60]
RATE_SUPRASPINAL_str=[ str(float(x)) for x in RATE_SUPRASPINAL ]
num_scs=0
NUM_SCS=num_scs
rate_scs= 40
num_MN=100
num_supraspinal=200
Name_fig="MNfr_supraspinal"
path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/"


#analysis firing rate vs num scs gamma distribution
# NUM_SCS=[1,5,10,12,14,16,18,20,30,40,50,60]
# Name_fig="MNfr_SCS_gamma_distribution"
# rate_scs= 40
# num_MN=100
# num_supraspinal=0
# rate_supraspinal=20
# RATE_SUPRASPINAL=rate_supraspinal



#analysis of MNs vs num_scs SMA with SMA sensory
##
# NUM_SCS=[1, 5, 10, 15, 20, 25, 30,40,50,60,70,80,90]
# Name_fig="MNfr_SCS"
# rate_scs= 40
# num_MN=100
# num_supraspinal=0
# rate_supraspinal=20
# RATE_SUPRASPINAL=rate_supraspinal


#path_simulations="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_SCS/"

mean_fr=[]
std_fr=[]
fr=[]
#path_file="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/WT100MNs_supraspinal5.0_SCSfreq40_SCS_num0.pickle"
#path_file="/Users/genis/SCS-SMA_Model_Results/Results/simulations/MNs_supraspinal_SCS/WT100MNs_supraspinal35.0_SCSfreq40_SCS_num0.pickle"
for irate, rate_supraspinal in enumerate(RATE_SUPRASPINAL_str): #analysis supraspinal
#for irate , num_scs in enumerate(NUM_SCS): #analysis SCS num
    # Name_file_record = MN_type + str(num_MN) + "MNs_supraspinal" + rate_supraspinal + "_SCSfreq" + str(
    #     rate_scs) + "_SCS_num" + str(num_scs) + ".pickle"

    # Name_file_record = MN_type + str(num_MN) + "MNs_supraspinal" + str(num_supraspinal) + "fr_" + str(
    #     rate_supraspinal) + "_SCSfreq" + str(rate_scs) + "_SCS_num" + str(num_scs) + ".pickle"

    #analysis SCS gamma distribution
    # Name_file_record = "GAMMA_distribution_" + MN_type + str(num_MN) + "MNs_supraspinal" + str(
    #     num_supraspinal) + "fr_" + str(rate_supraspinal) + "_SCSfreq" + str(rate_scs) + "_SCS_num" + str(
    #     num_scs) + ".pickle"

    Name_file_record="Only_supraspinal"+MN_type+str(num_MN)+"_supraspinal"+str(num_supraspinal)+\
                     "fr_"+str(rate_supraspinal)+"_SCSfreq"+str(rate_scs)+"_SCS_num"+str(num_scs)+".pickle"


    # Name_file_record = "Only_SCS" + MN_type + str(num_MN) + "_supraspinal" + str(num_supraspinal) + "fr_" + str(
    #     rate_supraspinal) + "_SCSfreq" + str(rate_scs) + "_SCS_num" + str(num_scs) + ".pickle"

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
    for ineuron in range(num_MN):
        #print(ineuron)
        x,t_spike_bins=tl.spike_times_2_spikes_window(data["MN_spikes"][ineuron], window, simulation_duration)
        spike_bins.append(x)

    spike_bins=np.array(spike_bins)
    t0=1000
    tf=simulation_duration
    it0=int(t0/window)
    itf=int(tf/window)
    fr.append( [1000*np.sum(spike_bins[ineuron][it0:itf])/(simulation_duration-t0) for ineuron in range(num_MN) ] ) #multiply by 1000 to have Hz

    mean_fr.append(np.mean(fr))
    std_fr.append(np.std(fr))


data={}

data["fr_MN"]=fr
data["rate_supraspinal"]=RATE_SUPRASPINAL
data["MN_type"]=MN_type
data["num_scs"]=NUM_SCS


f=open(path_simulations+Name_fig+"MN_type"+MN_type+".pickle","wb")
pickle.dump(data,f)
f.close()

#spike_bin=[spike_times_2_spikes_window(data["MN_spikes"][ineuron],window,simulation_duration) for ineuron in range(num_MN) ]