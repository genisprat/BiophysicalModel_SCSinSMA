
from neuron import h
import numpy as np


def create_input_neurons(N, rate, noise,first_spike=0):
    supraspinal_neurons = []
    if type(rate) == np.ndarray:
        for i in range(N):
            if rate[i]==0:
                cell = h.NetStim()
                cell.interval = 1000.0    # Inter-spike interval in ms
                cell.noise = noise
                cell.number = 1e999
                cell.start = 1e999
                # nc = h.NetCon(syn, None)
                supraspinal_neurons.append(cell)
            else:
                cell = h.NetStim()
                cell.interval = 1000.0/ rate[i]   # Inter-spike interval in ms
                cell.noise = noise
                cell.number = 1e999
                cell.start = first_spike
                # nc = h.NetCon(syn, None)
                supraspinal_neurons.append(cell)

    else:
        for _ in range(N):
            cell = h.NetStim()
            cell.interval = 1000.0 / rate  # Inter-spike interval in ms
            cell.noise = noise
            cell.number = 1e999
            cell.start = first_spike
            # nc = h.NetCon(syn, None)
            supraspinal_neurons.append(cell)
    return supraspinal_neurons


def create_spike_recorder_input_neurons(neurons):
    num_neurons = len(neurons)
    spike_times = [h.Vector() for i in range(num_neurons)]
    spike_detector = [h.NetCon(neurons[i], None) for i in range(num_neurons)]
    for i in range(num_neurons): spike_detector[i].record(spike_times[i])
    return spike_times


def create_spike_recorder_MNs(neurons):
    MN_spike_times = [h.Vector() for i in range(len(neurons))]
    MN_spike_detector = []
    for i in range(len(neurons)):
        sp_detector = h.NetCon(neurons[i].soma(0.5)._ref_v, None, sec=neurons[i].soma)
        sp_detector.threshold = -5
        MN_spike_detector.append(sp_detector)
        MN_spike_detector[i].record(MN_spike_times[i])
    return MN_spike_times

def create_exponential_synapses(source,target,W,tau,delay=0):
    syn_list=[]
    nc_list=[]

    for itarget in range(len(target)):
        syn_list.append([])
        nc_list.append([])

        for isource in range(len(source)):
            syn_ = h.ExpSyn(target[itarget].soma(0.5))
            syn_.tau = tau
            nc = h.NetCon(source[isource], syn_)
            nc.weight[0] = W[isource,itarget]
            nc_list[-1].append(nc)
            syn_list[-1].append(syn_)
            if type(delay) == np.ndarray: nc.delay=delay[isource,itarget]

    return syn_list,nc_list

