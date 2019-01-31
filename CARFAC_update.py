# -*- coding: utf-8 -*-


"""
Created on Thu Aug 1 2013

@author: andrevanschaik
"""

from pylab import *
from scipy import signal
from time import sleep
import colorsys
from mpl_toolkits.mplot3d import Axes3D
ion()

def COCHLEA(stimulus, npoints, fs, nsec, xlow, xhigh):

    
    #%%
    # BM parameters
    x = linspace(xhigh,xlow,nsec) # position along the cochlea 1 = base, 0 = apex
    f = 165.4*(10**(2.1*x)-1)     # Greenwood for humans
    a0 = cos(2*pi*f/fs)           # a0 and c0 control the poles and zeros
    c0 = sin(2*pi*f/fs)
    
    damping = 0.6                # damping factor
    #r = 1 - damping*2*pi*f/fs    # pole & zero radius actual
    r1 = 1 - damping*2*pi*f/fs    # pole & zero radius minimum (set point)
    h = c0                       # p279 h=c0 puts the zeros 1/2 octave above poles
    #h=0.5*(2+2*a0)/c0            # see Fig 16.1 and p279
    
    
    f_hpf = 20                    # p302 20Hz corner for the BM HPF
    q = 1/(1+(2*pi*f_hpf/fs))     # corresponding IIR coefficient 
    
    # IHC parameters
    u = 10 + 0.175                # p302 NLF function 
    u_max = u**3/(u**3+u**2+0.1)  # p303
    u_min = 0 + 0.175
    u_min = u_min**3/(u_min**3+u_min**2+0.1)
    tau_in = 10e-3                # p303 transmitter creation time constant
    tau_out = 0.5e-3              # p303 transmitter depletion time constant
    ro = 1/u_max                  # output resistance at a very high level
    c = tau_out/ro
    ri = tau_in/c
    saturation_output = 1/(2*ro + ri) # to get steady-state average, double ro for 50% duty cycle 
    r0 = 1/u_min                  # also consider the zero-signal equilibrium
    current = 1/(ri+r0)
    cap_voltage = 1-current*ri
    c_in = 1/(tau_in*fs)          # p303 corresponding IIR coefficient    
    c_out = ro/(tau_out*fs)       # p303 corresponding IIR coefficient 
    tau_IHC = 80e-6               # p303 ~2kHz LPF for IHC output
    c_IHC = 1/(fs*tau_IHC)        # corresponding IIR coefficient 
    output_gain = 1/(saturation_output - current)
    rest_output = current/(saturation_output - current)
    
    
    # OHC parameters
    scale = 0.1                   # p288
    offset = 0.04                 # p288 
    b = zeros(nsec)               # automatic gain loop feedback (1=no undamping).
    d_rz = 0.7*(damping*2*pi*f/fs)# p286,OHC gain 
    r = r1 + d_rz
    #r = 1 - damping*2*pi*f/fs 
    #r = 1 - damping*2*pi*f/fs
    g = (1-2*a0*r+r*r)/(1-(2*a0-h*c0)*r+r*r)  # p279 this gives 0dB DC gain for BM
    #g =1- (0.5*r)
    # AGC loop parameters
    total_DC_gain = 15.0          # p311, sum up of gain from each stage to nect slower stage 
    n_stage = 4
    tau = [0.002,0.008,0.032,0.128] 
    decim = [8,16,32,64]          # decimaiton 
    AGC1_scale = (1.0,1.4142,2.0,2.8284) # AGC1 pass is spatial smoothing from base to apex
    AGC2_scale = (1.65,2.3335,3.3,4.6669)# AGC2 pass is back
    ntimes = zeros(n_stage)       # effective number of smoothing in a time constant
    delay = zeros(n_stage)        # decide on target spread(variance) and delay(mean) of impulse response as a distribution to be convolved ntimes
    spread_sq = zeros(n_stage)
    # spatial filtering
    for m in range (n_stage):
        ntimes[m] = tau[m]*(fs/decim[m])
        delay[m] = (AGC2_scale[m] - AGC1_scale[m])/ntimes[m]
        spread_sq[m] = (AGC1_scale[m]**2 + AGC2_scale[m]**2)/ ntimes[m]
    
    sa = (spread_sq + delay*delay - delay) / 2;
    sb = (spread_sq + delay*delay + delay) / 2;
    sc = 1 - sa - sb
    
    # temporal
    tau_AGC1 = .128
    tau_AGC2 = .032
    tau_AGC3 = .008
    tau_AGC4 = .002
    
    # The AGC filters are decimated, i.e., running at a lower sample rate
    c_AGC1 = 64/(fs*tau_AGC1)
    c_AGC2 = 32/(fs*tau_AGC2)
    c_AGC3 = 16/(fs*tau_AGC3)
    c_AGC4 =  8/(fs*tau_AGC4)
    
    
    W0 = zeros(nsec)                          # BM filter internal state
    W1 = zeros(nsec)                          # BM filter internal state
    W1old = zeros(nsec)                       # BM filter internal state at t-1
    BM = zeros((nsec,npoints))                # BM displacement
    BM_1 = zeros((nsec,npoints))                # BM displacement
    BM_hpf = zeros((nsec,npoints))            # BM displacement high-pass filtered at 20Hz
    trans = cap_voltage*ones(nsec)            # transmitter available
    IHC = rest_output*ones((nsec,npoints))    # IHC internal state
    IHCa = rest_output*ones((nsec,npoints))   # IHC filter internal state
    IHC_out = zeros((nsec,npoints))           # IHC output
    In8 = zeros(nsec)                         # Accumulator for ACG4
    In16 = zeros(nsec)                        # Accumulator for AGC3
    In32 = zeros(nsec)                        # Accumulator for AGC2
    In64 = zeros(nsec)                        # Accumulator for AGC1
    AGC = zeros((nsec,npoints))               # AGC filter internal state
    AGC1 = zeros(nsec)                        # AGC filter internal state
    AGC2 = zeros(nsec)                        # AGC filter internal state
    AGC3 = zeros(nsec)                        # AGC filter internal state
    AGC4 = zeros(nsec)                        # AGC filter internal state
    v_ohc = zeros((nsec,npoints))
    nlf =  zeros((nsec,npoints))
    #%% play through cochlea
    BM[-1] = stimulus             # put stimulus at BM[-1] to provide input to BM[0]
    BM[-1,-1] = 0                 # hack to make BM_hpf[nsec-1,0] work
    for t in range(npoints):   
        for s in range(nsec):     # multiplex through the sections to calculate BM filters
            W0new = BM[s-1,t] + r[s]*(a0[s]*W0[s] - c0[s]*W1[s])
            W1[s] = r[s]*(a0[s]*W1[s] + c0[s]*W0[s])
            W0[s] = W0new
            BM[s,t] = g[s]*(BM[s-1,t] + h[s]*W1[s])
            #BM[s,t] = (BM[s-1,t] + h[s]*W1[s])
            #BM_hpf[s,t] = BM[s,t] - BM[s-1,t]
        # to speed up simulation, operate on all sections simultaneously for what follows               
        BM_hpf[:,t] = q*(BM_hpf[:,t-1] + BM[:,t] - BM[:,t-1])   # high-pass filter    
        z_int = (1 - (BM_hpf[:,t]+0.13)/4).clip(0)
        z = ((z_int)**8).clip(0,1)                              # nonlinear function for IHC
        v_mem = 0.75*((1-z)**2)                                 # IHC membrane potential
        IHC_new = v_mem*trans                                   # IHC output
        trans += c_in*(1-trans) - c_out*IHC_new                 # update amount of neuro transmitter
        IHC_new = IHC_new * output_gain
        IHCa[:,t] = (1-c_IHC)*IHCa[:,t-1] + c_IHC*IHC_new       # Low-pass filter once
        IHC[:,t] = (1-c_IHC)*IHC[:,t-1] + c_IHC*IHCa[:,t]       # Low-pass filter twice  
        IHC_out[:,t] = IHC[:,t] - rest_output
        v_OHC = W1[:] - W1old[:]                                      # OHC potential 
        #v_ohc[:,t] = W1[:] - W1old[:]    
        W1old[:] = W1[:]    
        sqr=(v_OHC*scale+offset)**2                                           # nonlinear function for OHC
        NLF=(1-sqr/8.0)**8    
        nlf[:,t] = NLF
        r = r1 + d_rz*(1-b)*NLF                                 # feedback to BM
        g = (1-2*a0*r+r*r)/(1-(2*a0-h*c0)*r+r*r) 
    
        In8 += IHC_out[:,t]/total_DC_gain                       # subsample AGC4 by factor total_DC_gain
        if t%8 == 0:
            In16 += In8
            In32 += In8
            In64 += In8
            if t%64 == 0:                                       # subsample AGC1 by factor 64
                AGC1 = (1-c_AGC1)*AGC1 + c_AGC1*In64/64.0       # LPF in time domain
                AGC1 = sa[3]*append(AGC1[0],AGC1[0:nsec-1]) + (1-sa[3]-sb[3])*AGC1 + sb[3]*append(AGC1[1:nsec],AGC1[-1]) # LPF in spatial domain
                In64 *= 0.0                                     # reset accumulator
            if t%32 == 0:                                       # subsample AGC2 by factor 32
                AGC2 = (1-c_AGC2)*AGC2 + c_AGC2*(In32/32.0 + 2*AGC1)
                AGC2 = sa[2]*append(AGC2[0],AGC2[0:nsec-1]) + (1-sa[2]-sb[2])*AGC2 + sb[2]*append(AGC2[1:nsec],AGC2[-1]) 
                In32 *= 0.0
            if t%16 == 0:                                       # subsample ACG3 by factor 16
                AGC3 = (1-c_AGC3)*AGC3 + c_AGC3*(In16/16.0 + 2*AGC2)
                AGC3 = sa[1]*append(AGC3[0],AGC3[0:nsec-1]) + (1-sa[1]-sb[1])*AGC3 + sb[1]*append(AGC3[1:nsec],AGC3[-1])  
                In16 *= 0.0
            AGC4 = (1-c_AGC4)*AGC4 + c_AGC4*(In8/8.0 + 2*AGC3)
            AGC4 = sa[0]*append(AGC4[0],AGC4[0:nsec-1]) + (1-sa[0]-sb[0])*AGC4 + sb[0]*append(AGC4[1:nsec],AGC4[-1]) 
            In8 *= 0.0
            AGC[:,t] = AGC4                                     # store AGC output for plotting
            b = AGC4                                            # arbitrary gain factor, not sure this is correct
            b = b.clip(0,1)                                     # limit b (damping) between 0 and 1

    kernel = [-0.33,1.66,-0.33]                   
    IHC_li = zeros((nsec,npoints))
    IHC_li_hpf = zeros((nsec,npoints))
    hpf_c = (1/(1+(2*pi*(4*f/5)/fs)))*ones(nsec)
    
    for s in range (nsec): 
       IHC_li[s,:] =  convolve(IHC[s,:], kernel, 'same')
    
    for s in range(nsec):  
        for t in range(npoints):
            IHC_li_hpf[s,t] = hpf_c[s]*(IHC_li_hpf[s,t-1] + (IHC_li[s,t]-IHC_li[s,t-1]));
        
    return BM_hpf,IHC_out;     
    
    
    
    
