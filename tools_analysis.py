
import glob
import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft,fftfreq
from sys import path
from importlib import reload

path.insert(1, '/Users/genis/Dropbox/python3_functions/')

import help_plot as hp
reload(hp)


def moving_average(x, w,s=-1):
    if s==-1:
        return np.convolve(x, np.ones(w), 'valid') / w
    else:
        window_avg = [ np.mean(x[int(i-w/2):int(i+w/2)]) for i in range(int(w/2), len(x), s) if i+w/2 <= len(x) ]
        times=[i for i in range(int(w/2), len(x), s) if i+w/2 <= len(x) ]
        return window_avg,times


def firing_rate(x,w,s):
    if w==1 and s ==1:
        return x,range(len(x))
    else:
        window_avg = [ np.mean(x[int(i-w/2):int(i+w/2)]) for i in range(int(w/2), len(x), s) if i+w/2 <= len(x) ]
        times=[i for i in range(int(w/2), len(x), s) if i+w/2 <= len(x) ]
    return window_avg,times

def cross_correlation_normalized(a1, a2,forward=True):
	if forward:
		lags=range(0,len(a2))
	else:
		lags = range(-len(a1)+1, len(a2))

	cs = []
	for lag in lags:
	    idx_lower_a1 = max(lag, 0)
	    idx_lower_a2 = max(-lag, 0)
	    idx_upper_a1 = min(len(a1), len(a1)+lag)
	    idx_upper_a2 = min(len(a2), len(a2)-lag)
	    b1 = a1[idx_lower_a1:idx_upper_a1]
	    b2 = a2[idx_lower_a2:idx_upper_a2]
	    c = np.corrcoef(b1, b2)[0,1]
	    #c = c / np.sqrt((b1**2).sum() * (b2**2).sum())
	    cs.append(c)
	return cs,lags



def spike_times(spikes,dt=1,axes=None,idneuron=0,color='k'):
    '''
    This function trasforms a vector  0 and 1 into a vector
    of spikes times to plot a raster. the units depend on the dt.
    dt should be in ms.

    '''
    T=len(spikes)
    spike_timess=dt*np.where(spikes==1)[0]
    #print(spike_times)
    if axes is not None:
        axes.plot(spike_timess,idneuron*np.ones(len(spike_timess)),".",color=color)

    return spike_timess



def inter_spikes_intervals(spikes_times):
    '''
    This function trasforms a vector  of spikes_times into a vector of  spikes times
    intervals.

    '''
    isi=np.diff(spikes_times)

    return isi


def inter_spikes_intervals_normalize(spikes,period):
    '''
    This function trasforms an array of spikes (NMN x T) to a vector of normalized spike_times.

    '''
    Nneurons,NT=np.shape(spikes)

    spike_times_aux=[ spike_times(spikes[ineuron]) for ineuron in range(Nneurons)]
    inter_spike_interval=np.concatenate([ inter_spikes_intervals(spike_times_aux[ineuron]) for ineuron in range(Nneurons)] )

    inter_spike_interval_norm=inter_spike_interval/period


    return inter_spike_interval_norm


def cross_correlation_histogram(spikes,ineuron=0,jneuron=1,bins=None,Nbins=1001):

    Nmn,T=np.shape(spikes)
    if bins is None:
        bins=np.linspace(-T,T,Nbins)



    #spikes_times=[ spike_times(spikes[i]) for i in range(Nmn) ]
    # diff_spike_times=np.array([ tispike - spikes_times[jneuron][jspikes] for ineuron in range(Nmn)
    #  for jneuron in range(ineuron+1,Nmn) for tispike in spikes_times[ineuron] for tjspike in spikes_times[jneuron]  ] )


    ispikes_times= spike_times(spikes[ineuron])
    jspikes_times= spike_times(spikes[jneuron])
    diff_spike_times=np.array([ tispike - tjspike for tispike in ispikes_times for tjspike in jspikes_times  ] )
    print(diff_spike_times)
    hist,_=np.histogram(diff_spike_times,bins=bins)

    bins_centers=(bins[1:] + bins[:-1])/2

    return hist,bins_centers


def firing_rate_to_force(spikes,Norm=None,seed=None):
    """
    Spikes to force folllowing Models of recruiment and rate coding organization
    in Motor Unit Pools. Fluglevant det al 1993
    Here we assume that the time bin is 1 ms
    """

    if seed is not None: np.random.seed(seed)
    Nmn,total_time=np.shape(spikes)

    F=np.zeros(total_time)

    #Peak twich force
    RP=100 #range of twich force
    b=np.log(RP)
    P=np.exp(np.array(range(1,Nmn+1))*b/Nmn)
    np.random.shuffle(P) # I shuffle because the other of the motoneurons is always first WT then SMA. and we do not make any assumptions about the force of each type

    #contraction times

    TL=90 ###this is ms###
    RT=3 ### Range of contraction times
    c=np.log(RP)/np.log(RT) # this is equivalent to log_RT(RP)
    T=TL/(P**(1/c))
    np.random.shuffle(T)
    #plt.figure()
    #plt.hist(T)

    #compute mean isi. First aproximation of the entire simuation
    sp_times=[spike_times(spikes[ineuron]) for ineuron in range(Nmn)]
    mean_isi=[np.mean( inter_spikes_intervals(sp_times[ineuron]) ) for ineuron in range(Nmn)]


    g=np.zeros(Nmn)
    f=[]

    print("putaaaaa")

    for ineuron in range(Nmn):
        t=np.array(range(int(10*T[ineuron])))
        #g[ineuron]=1
        if len(sp_times[ineuron]==0):
            g[ineuron]=1

        elif T[ineuron]/mean_isi[ineuron]<0.4 :
            g[ineuron]=1
        else:
            aux=T[ineuron]/mean_isi[ineuron]
            s=1-np.exp(-2*(aux)**3) #eq 16
            g[ineuron]=s/aux #eq 17


        f.append( (g[ineuron]*P[ineuron]*t/T[ineuron])*np.exp(1-t/T[ineuron]) )




    for ineuron in range(Nmn):
        spike_index=np.where(spikes[ineuron]==1)[0]
        #print(len(spike_index))
        for i_initial in spike_index:
            i_final=min(i_initial+len(f[ineuron]),total_time)
            #print(i_initial,i_final)
            F[i_initial:i_final]=F[i_initial:i_final]+np.array(f[ineuron][:i_final-i_initial])


    if Norm is not None:
        F=F/Norm

    return F







def amplitudes_frequencies(path):
    frequencies=[]
    amplitudes=[]
    ij=0
    for filename in glob.glob(path+'*.p'):
        ij=ij+1
        print(filename)
        file=open(filename,'rb')
        data=pickle.load(file)
        try:
            f=round(data["frequency"],2)
            a=round(data["ampllitude"][0],2)
        except KeyError:
            f=round(data["scsfrequency"],2)
            a=round(data["scsampllitude"][0],2)

        if f  not in frequencies:
            frequencies.append(f)
        if a not in amplitudes:
            amplitudes.append( a)

    print(ij)

    return np.sort(amplitudes),np.sort(frequencies)



def fft_frmn(fr_mn,step):
    N=len(fr_mn)
    T=step/1000 #so seconds

    yf=fft(fr_mn-np.mean(fr_mn))
    # I remove the frequency=0 because it is only a bias.
    #It means that the mean fr is not zero
    #Alternatively, we can remove the mean of the signal.
    power=2.0/N * np.abs(yf[0:N//2])

    xf = fftfreq(N, T)[:N//2]
    return power,xf


def plot_cst_force(axes,fname,scs,cst,spatiotmep,freq,max_force,amplitude,inital_t,final_t,window_av,step,color,line_style,xticks):
    f=open(fname,"rb")
    data=pickle.load(f)
    f.close()
    _,_,N_MN,T=np.shape(data["mSpikes_all"])
    spikes_mn_all=data["mSpikes_all"][0][0]
    spikes_MN=data["mSpikes"][0]

    fr_mn,times=firing_rate(spikes_MN,window_av,step)
    fr_mn=np.array(fr_mn)*1000/N_MN #spikes_MN contains the spikes of all neurons in ms time bins. Transform spikes/ms to spikes/(neuron s)

    if cst:
        _,_,N_CST,T=np.shape(data["mSpikes_CM_all"])

        spikes_CST=data["mSpikes_CM"][0]

        fr_cst,times=firing_rate(spikes_CST,window_av,step)
        #mSpikes_CST contains the spikes of all CST neurons in ms time bins. Transform spikes/ms to spikes/(s), number of input spikes per MN
        fr_cst=np.array(fr_cst)*1000 #mSpikes_CST contains the spikes of all CST neurons in ms time bins. Transform spikes/ms to spikes/(s)

        #fr_cst=np.array(fr_cst)/N_CST #mSpikes_CM contains the spikes of all neurons in ms time bins. Transform spikes/ms to spikes/(neuron s)
        fr_cst=np.array(fr_cst)


    else:
        fr_cst=np.zeros(len(fr_mn))

    if not scs:
        fr_Iaf=np.zeros(len(fr_cst))
        Iaf_activated=np.zeros(len(data["mSpikes_CM"][0]))
        times_Iaf=np.linspace(times[0],times[-1],num=len(fr_Iaf))

    else:
        Iaf_activated=data["mSpikes_Iaf"][0]
        times_Iaf_Activated=range(0, len(Iaf_activated))
        fr_Iaf,times_Iaf=np.array(firing_rate(Iaf_activated,window_av,step))

        fr_Iaf=np.ones(len(times))*(amplitude/100.0)
        if spatiotmep==True:
            Tpulse=200
            aux=np.concatenate((np.ones(Tpulse),np.zeros(Tpulse)))
            fr_Iaf=(amplitude/100.0)*np.concatenate((aux,aux,aux))
        times_Iaf=np.linspace(times[0],times[-1],num=len(fr_Iaf))
    # plot Firing rates




    axes[1].plot(times,fr_mn,line_style[1],color=color,lw=2)
    axes[2].plot(times,fr_cst,line_style[2],color=color,lw=2)


    fontsize=8

    for i in range(len(axes)): axes[i].set_xlim(inital_t,final_t)
    hp.yticks(axes[1],[0,15,30],["","",""],fontsize=fontsize)
    hp.xticks(axes[1],xticks,len(xticks)*[""],fontsize=fontsize)
    axes[1].set_ylim(-2,30)


    hp.yticks(axes[2],[0,4000,8000],["","",""],fontsize=fontsize)
    #hp.yticks(axes[2],[0,30,60],["0","30","60"],fontsize=fontsize)
    hp.xticks(axes[2],xticks,len(xticks)*[""],fontsize=fontsize)


    hp.xticks(axes[2],xticks,len(xticks)*[""],fontsize=fontsize)


    hp.remove_axis(axes[0])
    hp.remove_axis(axes[1])
    hp.remove_axis(axes[2])


    F=firing_rate_to_force(spikes_mn_all,max_force)
    #F=firing_rate_to_force(spikes_mn_all)

    F_ave,times=moving_average(F,window_av,step)

    axes[0].plot(times,F_ave,line_style[0],color=color)

    #axes[0].set_ylabel("Force/Max force")
    hp.xticks(axes[0],xticks,len(xticks)*[""],fontsize=fontsize)
    hp.yticks(axes[0],[0,0.5,1],["","",""],fontsize=fontsize)
    axes[0].set_ylim([-0.1,1.3])

    #for i in range(len(axes)):axes[i].set_xlim([0,final_t])

    return times,fr_Iaf,fr_cst,fr_mn,F_ave


def plot_scs_cst_force(axes,fname,scs,cst,spatiotmep,freq,max_force,amplitude,inital_t,final_t,window_av,step,color,line_style,xticks):
    f=open(fname,"rb")
    data=pickle.load(f)
    f.close()
    _,_,N_MN,T=np.shape(data["mSpikes_all"])
    spikes_mn_all=data["mSpikes_all"][0][0]
    spikes_MN=data["mSpikes"][0]

    fr_mn,times=firing_rate(spikes_MN,window_av,step)
    fr_mn=np.array(fr_mn)*1000/N_MN #spikes_MN contains the spikes of all neurons in ms time bins. Transform spikes/ms to spikes/(neuron s)

    if cst:
        _,_,N_CST,T=np.shape(data["mSpikes_CM_all"])

        spikes_CST=data["mSpikes_CM"][0]

        fr_cst,times=firing_rate(spikes_CST,window_av,step)
        #mSpikes_CST contains the spikes of all CST neurons in ms time bins. Transform spikes/ms to spikes/(s), number of input spikes per MN
        fr_cst=np.array(fr_cst)*1000 #mSpikes_CST contains the spikes of all CST neurons in ms time bins. Transform spikes/ms to spikes/(s)

        #fr_cst=np.array(fr_cst)/N_CST #mSpikes_CM contains the spikes of all neurons in ms time bins. Transform spikes/ms to spikes/(neuron s)
        fr_cst=np.array(fr_cst)


    else:
        fr_cst=np.zeros(len(fr_mn))

    if not scs:
        fr_Iaf=np.zeros(len(fr_cst))
        Iaf_activated=np.zeros(len(data["mSpikes_CM"][0]))
        times_Iaf=np.linspace(times[0],times[-1],num=len(fr_Iaf))

    else:
        Iaf_activated=data["mSpikes_Iaf"][0]
        times_Iaf_Activated=range(0, len(Iaf_activated))
        fr_Iaf,times_Iaf=np.array(firing_rate(Iaf_activated,window_av,step))

        fr_Iaf=np.ones(len(times))*(amplitude/100.0)
        if spatiotmep==True:
            Tpulse=200
            aux=np.concatenate((np.ones(Tpulse),np.zeros(Tpulse)))
            fr_Iaf=(amplitude/100.0)*np.concatenate((aux,aux,aux))
        times_Iaf=np.linspace(times[0],times[-1],num=len(fr_Iaf))
    # plot Firing rates

    iplot_force=0
    iplot_MN=1
    iplot_cst=2


    axes[1].plot(times,fr_mn,line_style[1],color=color,lw=2)
    axes[2].plot(times,fr_cst,line_style[2],color=color,lw=2)
    axes[3].plot(times_Iaf,fr_Iaf,line_style[3],color=color,lw=2)


    fontsize=8

    for i in range(len(axes)): axes[i].set_xlim(inital_t,final_t)
    hp.yticks(axes[1],[0,15,30],["","",""],fontsize=fontsize)
    hp.xticks(axes[1],xticks,len(xticks)*[""],fontsize=fontsize)
    axes[1].set_ylim(-2,30)


    hp.yticks(axes[2],[0,4000,8000],["","",""],fontsize=fontsize)
    #hp.yticks(axes[2],[0,30,60],["0","30","60"],fontsize=fontsize)
    axes[3].set_ylim(-500,8000)
    hp.xticks(axes[2],xticks,len(xticks)*[""],fontsize=fontsize)


    hp.yticks(axes[3],[0,0.5,1.0],["","",""],fontsize=fontsize)
    axes[3].set_ylim(-0.1,1.0)
    hp.xticks(axes[3],xticks,len(xticks)*[""],fontsize=fontsize)


    hp.remove_axis(axes[0])
    hp.remove_axis(axes[1])
    hp.remove_axis(axes[2])


    F=firing_rate_to_force(spikes_mn_all,max_force)
    #F=firing_rate_to_force(spikes_mn_all)

    F_ave,times=moving_average(F,window_av,step)

    axes[0].plot(times,F_ave,line_style[0],color=color)

    #axes[0].set_ylabel("Force/Max force")
    hp.xticks(axes[0],xticks,len(xticks)*[""],fontsize=fontsize)
    hp.yticks(axes[0],[0,0.5,1],["","",""],fontsize=fontsize)
    axes[0].set_ylim([-0.1,1.3])

    #for i in range(len(axes)):axes[i].set_xlim([0,final_t])

    return times,fr_Iaf,fr_cst,fr_mn,F_ave


def spike_times_2_spikes_window(spikes_times,window,simulation_duration):
    if simulation_duration % window==0:
        Nbins = int(simulation_duration / window)
        t_spikes_bin = np.arange(window / 2, Nbins * window, window)
    else:
        Nbins = int(simulation_duration / window)+1
        t_spikes_bin = np.arange(window / 2, Nbins * window, window)[:-1]
    spikes_bin=np.zeros(Nbins)
    for ispike in range(len(spikes_times)):
        spikes_bin[int(spikes_times[ispike]/window)]+=1

    return spikes_bin,t_spikes_bin


def dumped_sinus_function(amplitude,tau,sin_f,spike_times,delay,duration,dt=0.1):
    y=np.zeros(int(duration/dt))
    t=np.linspace(0,duration,int(duration/dt))
    for spike_time in spike_times:
        t0=spike_time+delay
        dumped_sin=amplitude*np.exp(-(t-t0)/tau)*np.sin(2*np.pi*sin_f*(t-t0))
        dumped_sin[t<spike_time+delay]=0
        dumped_sin[t < spike_time + delay] = 0
        y=y+dumped_sin
    return t,y






