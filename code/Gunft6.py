# -*- coding: utf-8 -*-


'''
Paper:Peter beim Graben and Serafim rodrigues
A Biophysical observation model for field potentials of networks of leaky-integrate-and-fire neurons.
Front. Comput. Neurosci, 04 January 2013 | doi: 10.3389/fncom.2012.00100

*******************************
This python code that runs under the Brian simulator.


Note for developers:
Note1 : As it stands, the code is not effiecient (fast) as it does not use the facilties vector processing and uses a lot of for-loops which is not efficient.
So it can be improved.
Note 2: Periodic thalamic input is not yet implemented.

*****************************

1) This is a network of 5000 neurons, 80% of which excitatory, and 
   20% inhibitory.
2) The network is randomy connected (between pairs) with connection 
   probability = 0.2.
3) Both Excitatory and Inhibitory neurons are described via LIF model.
4) The currents are double exponetial, but the excitatory currents
   can recieve external noise.

'''
from brian2 import *
from brian2.input import PoissonThreshold

from math import *
from sys import *
import numpy as np
import matplotlib.pyplot as plt
import statistics as st  # Assuming you have already imported the 'statistics' module



#General parameters
setup_duration = 1750*ms # initial run to settle behaviour
duration = 250*ms  # Run time for recording
#duration = 2000*ms
defaultclock.dt = 0.01 * ms


#----------------------------- GLOBAL Parameters ----------------------------------------------------------
tau_mE = 20*ms    # membrane time constant for excitatory [pyramidal] neurons
tau_mI = 10*ms    # membrane time constant for inhibitory neurons
tau_rG_I = 0.25*ms  # rise time of GABA synaptic current (interneurons)
tau_dG_I = 5*ms     # decay time of GABA synaptic current (interneurons)
tau_rG_E = 0.25*ms  # rise time of GABA synaptic current (pyramidal cells)
tau_dG_E = 5*ms     # decay time of GABA synaptic current (pyramidal cells)
tau_rA_I = 0.2*ms # rise time of AMPA synpatic current (on interneurons)
tau_dA_I = 1*ms   # decay time of AMPA synpatic current (on interneurons)
tau_rA_E = 0.4*ms # rise time of AMPA synpatic current (pyramidal [pyr] cells)
tau_dA_E = 2*ms   # decay time of AMPA synpatic current (pyramiday[pyr] cells)
tau_rp_E = 2*ms   # refractory period for Pyramidal cells
tau_rp_I = 1*ms   # refractory period of Inhibitory cells
Vt = 18*mV        # spike threshold
Vreset = 11*mV    # Reset value
El = 0 * mV       # resting potential

# Latency of Post Synaptic Potential
tau_EL = 1*ms # For exctiatory neurons
tau_IL = 1*ms # For inhibitory neurons
tau_TL = 1*ms # For thalamic input


J_GABA_E = 1.7*mV  # Gaba synaptic efficacy on pyramidal cells
J_GABA_I = 2.7*mV  # Gaba synaptic efficacy on interneurons
J_AMPA_E = 0.42*mV # Ampa synaptic efficacy on pyramidal cells
J_AMPA_I = 0.7*mV  # Ampa synaptic efficacy om Interneurons
J_Ext_E = 0.55*mV  # External/Thalamic synpatic efficacy on pyramidal cells
J_Ext_I = 0.95*mV  # External/Thalamic synaptic efficacy on interneurons



Number_Of_CorticalNeurons = 5000 # Total number of Cortiucal Neurons
##P_Interneurons = 0.2              # 20% Percent of Interneurons
##P_Excitatory = 1 - P_Interneurons # Percentage of Excitatory neurons
##N_Interneurons = int(P_Interneurons*Number_Of_CorticalNeurons)
##N_Excitatory = int(P_Excitatory*Number_Of_CorticalNeurons)

N_Interneurons = 1000
N_Excitatory = 4000
Number_Of_ThalamicNeurons = 5000 # Thalamic/external Input (every neuron will recieve a different external realisation)




#-----------------------------Parameters  ------------------------------------------------------

r_i = 1.7e9 
Rho_Cytoplasm = 2
Rho_Extracellular = 3.33  
Rho_Membrane_Hillock  = 0.005 
Length_Dendrite = 2e-4 
Radius_Dendrite = 7e-6
Length_Axon_Hillock = 2e-5
Radius_Axon_Hillock = 5e-7
Area_Axon_Hillock = (2*pi*(Radius_Axon_Hillock)**2 + 2*pi*Radius_Axon_Hillock*Length_Axon_Hillock)
Area_Extracellular = ((12*sqrt(3) - 3*pi)*(Radius_Dendrite)**2)
R_A = Rho_Cytoplasm*(Length_Dendrite/2)/(pi*Radius_Dendrite**2) 
R_B = Rho_Cytoplasm*(Length_Dendrite/2)/(pi*Radius_Dendrite**2)
R_C = Rho_Extracellular*(Length_Dendrite/2)/Area_Extracellular
R_D = Rho_Extracellular*(Length_Dendrite/2)/Area_Extracellular
R_M = Rho_Membrane_Hillock/Area_Axon_Hillock
# Print the calculated resistances and areas for verification
print("Area_Axon_Hillock:", Area_Axon_Hillock)
print("Area_Extracellular:", Area_Extracellular)
print("R_A:", R_A)
print("R_B:", R_B)
print("R_C:", R_C)
print("R_D:", R_D)
print("R_M:", R_M)



#----------- Setup Cortical Model ----------------------------
ENeuronModel = Equations('''
        dV/dt  = (-(V-El) + IA - IG)/tau_mE : volt
        dIA/dt = (-IA + XA)/tau_dA_E         : volt
        dXA/dt = -XA/tau_rA_E                : volt
        dIG/dt = (-IG + XG)/tau_dG_E         : volt
        dXG/dt = -XG/tau_rG_E                : volt
        dIA2/dt = (-IA2 + XA2)/tau_dA_E      : volt
        dXA2/dt = -XA2/tau_rA_E              : volt
        dIG2/dt = (-IG2 + XG2)/tau_dG_E      : volt
        dXG2/dt = -XG2/tau_rG_E              : volt
        LFP = abs(IA) + abs(IG)              : volt
        DFP = IA2 + IG2 + PSI*V              : volt
        PSI                                  : 1
        ''')

INeuronModel = Equations('''
        dV/dt  = (-(V-El) + IA - IG)/tau_mI : volt
        dIA/dt = (-IA + XA)/tau_dA_I         : volt
        dXA/dt = -XA/tau_rA_I                : volt
        dIG/dt = (-IG + XG)/tau_dG_I         : volt
        dXG/dt = -XG/tau_rG_I                : volt
        LFP = abs(IA) + abs(IG)              : volt
        ''')
Ge = NeuronGroup(N_Excitatory, model=ENeuronModel, reset=Vreset, threshold=Vt, refractory=tau_rp_E) # Excitatory neurons
Gi = NeuronGroup(N_Interneurons, model=INeuronModel, reset=Vreset, threshold=Vt, refractory=tau_rp_I) # Inhibitory neurons


'''
And in order to start the network off in a somewhat
more realistic state, we initialise the membrane
potentials uniformly randomly between the reset and
the threshold.
'''
Ge.V = Vreset + (Vt - Vreset) * rand(len(Ge))
Gi.V = Vreset + (Vt - Vreset) * rand(len(Gi))
Ge.IA = zeros(len(Ge))
Gi.IA = zeros(len(Gi))
Ge.XA = zeros(len(Ge))
Gi.XA = zeros(len(Gi))
Ge.IG = zeros(len(Ge))
Gi.IG = zeros(len(Gi))
Ge.XG = zeros(len(Ge))
Gi.XG = zeros(len(Gi))
Ge.LFP = zeros(len(Ge))
Gi.LFP = zeros(len(Gi))
Ge.DFP = zeros(len(Ge))

       
#------------------------- Setup Thalamic Input ----------------------------
''' 
The Thalamic input is essential noise. The idea is to account for both cortical Heterogeneity and Spontaneous actvity.
For this reason we have two levels of noise. The first one is a an Ornstein-Uhlenbeck Process + (either constant or periodic signal)
with rate(t). The second level is time varying Inhomogenous Poisson process,with rate(t)
'''
# Ornstein-Uhlenbeck Process Parameters
tau_n = 16 * ms # Auto-Correlation time constant of the noise
st_dev = 400*Hz # Standard deviation [spikes/ms]

# Periodic signal parameters
##v_0 = 2400*Hz    # Base line of the periodic signal
##a = 400*Hz      # Amplitude of the periodic signal
##omega = 4 *Hz   # Angualr Frequency

v_0 = 1600*Hz
a = 400*Hz
omega = 10 *Hz   # Angualr Frequency

# Thalamic model
# This version is constant signal + noise
##OU_Constant = Equations(''' 
##dn/dt = -n/tau_n + st_dev*(2./tau_n)**.5*xi :Hz 
##rate = v_0 + n :Hz''')

# This version is periodic signal + noise
OU_Periodic = Equations(''' 
dn/dt = -n/tau_n + st_dev*(2./tau_n)**.5*xi :Hz 
signal = v_0 + a*sin(2*pi*omega*t) :Hz
rate = signal + n : Hz
''')


# This group does not produce spikes, and when negative rate, it will be considred as 0
# by the Poisson threshold on the seocond noise level 
#OU = NeuronGroup(1, model=OU_Constant)

OU = NeuronGroup(1, model=OU_Periodic)

# Second level noise
# Run N_Neurons realisations of inhomogenous Poisson process with rate, rate(t) given
# by the Ornstein_Uhlenbeck process.
Thalamic_IP = NeuronGroup(Number_Of_ThalamicNeurons, model='P : Hz', threshold=PoissonThreshold(state='P'))
#  IP.P = rand(N_Neurons) no need for this...
Thalamic_IP.P = linked_var(OU, 'rate')



#define network connections and weights
Wii = (tau_mI * J_GABA_I) / tau_rG_I
Cii = Synapses(Gi, Gi, model='XG', on_pre='XG += Wii', delay=tau_IL)
Cii.connect(p=0.2)

Wei = (tau_mI * J_AMPA_I) / tau_rA_I
Cei = Synapses(Ge, Gi, model='XA', on_pre='XA += Wei', delay=tau_EL)
Cei.connect(p=0.2)

Wie = (tau_mE * J_GABA_E) / tau_rG_E
Cie = Synapses(Gi, Ge, model='XG', on_pre='XG += Wie', delay=tau_IL)
Cie.connect(p=0.2)

Wee = (tau_mE * J_AMPA_E) / tau_rA_E
Cee = Synapses(Ge, Ge, model='XA', on_pre='XA += Wee', delay=tau_EL)
Cee.connect(p=0.2)

Wti = (tau_mI * J_Ext_I) / tau_rA_I
Cti = Synapses(Thalamic_IP, Gi, model='XA', on_pre='XA += Wti', delay=tau_TL)
Cti.connect(p='i >= N_Excitatory', skip_if_invalid=True)

Wte = (tau_mE * J_Ext_E) / tau_rA_E
Cte = Synapses(Thalamic_IP, Ge, model='XA', on_pre='XA += Wte', delay=tau_TL)
Cte.connect(p='i < N_Excitatory', skip_if_invalid=True)

alpha_list = [0.32 * 1e-9] * N_Excitatory
gI_list = [0] * N_Excitatory
gE_list = [0] * N_Excitatory

for i in range(N_Excitatory):
    for j in range(N_Excitatory):
        if Cee.is_connected(j, i):
            alpha_list[i] = alpha_list[i] + 0.25 * 1e-9  # 0.25*nS + 0.32*nS (cortical + thalamic)
    for k in range(N_Interneurons):
        if Cie.is_connected(k, i):
            gI_list[i] = gI_list[i] + 1 * 1e-9  # gaba = 1nS



# Evaluate other parameters
gE_list = [alpha_list[i] / (1 - (R_A + R_D) * alpha_list[i]) for i in range(N_Excitatory)]
   




We_tilda = [R_D * (J_AMPA_E) * (1 / r_i - gE_list[i] * (R_B + R_C) / (r_i * (1 + gE_list[i] * (R_A + R_D) + (R_B + R_C) * (gE_list[i] - gI_list[i] * (1 + gE_list[i] * (R_A + R_D)))))) for i in range(N_Excitatory)]
Wi_tilda = [R_D * (J_GABA_E) * (gE_list[i] * (R_B + R_C) / (r_i * (1 + gE_list[i] * (R_A + R_D) + (R_B + R_C) * (gE_list[i] - gI_list[i] * (1 + gE_list[i] * (R_A + R_D)))))) for i in range(N_Excitatory)]

#------ Now establish connections with \tilda{w}------
# Inhibitory-Excitatory
Cie2 = Synapses(Gi, Ge, model='XG2', on_pre='XG2 += Wi_tilda[j]', delay=tau_IL)
for i in range(N_Excitatory):
    for j in range(N_Interneurons):
        if Cie.is_connected(j, i):
            Cie2.connect(j=j, i=i)
            
Cee2 = Synapses(Ge, Ge, model='XA2', on_pre='XA2 += We_tilda[j]', delay=tau_EL)
for i in range(N_Excitatory):
    for j in range(N_Excitatory):
        if Cee.is_connected(j, i):
            Cee2.connect(j=j, i=i)

Cte2 = Synapses(Thalamic_IP, Ge, model='XA2', on_pre='XA2 += We_tilda[j]', delay=tau_TL)
for i in range(N_Excitatory):
    for j in range(N_Excitatory):
        if Cte.is_connected(j, i):
            Cte2.connect(j=j, i=i)


#-------------------------------------------------------------------





# Run the initial setup to settle behavior
run(setup_duration)

# Set up monitors
ThalamicRate = StateMonitor(Thalamic_IP, 'P', record=0, timestep=10)
TS = SpikeMonitor(Thalamic_IP)
Ge_S = SpikeMonitor(Ge[0:200])
Gi_S = SpikeMonitor(Gi[0:200])
LFP_e = StateMonitor(Ge, 'LFP', timestep=10)
DFP_e = StateMonitor(Ge, 'DFP', timestep=10)
IA2_e = StateMonitor(Ge, 'IA2', timestep=10)
IG2_e = StateMonitor(Ge, 'IG2', timestep=10)
IA_e = StateMonitor(Ge, 'IA', timestep=10)
IG_e = StateMonitor(Ge, 'IG', timestep=10)
Membrane = StateMonitor(Ge, 'V', timestep=10)
rate_i = PopulationRateMonitor(Gi, bin=1 * ms)
rate_e = PopulationRateMonitor(Ge, bin=1 * ms)

# Run the main simulation
run(duration)
'''



And finally we plot the results. Just for fun, we do a rather more
complicated plot than we've been doing so far, with three subplots.
The upper one is the raster plot of the whole network, and the
lower two are the values of ``V`` (on the left) and ``ge`` and ``gi`` (on the
right) for the neuron we recorded from. See the PyLab documentation
for an explanation of the plotting functions, but note that the
:func:`raster_plot` keyword ``newfigure=False`` instructs the (Brian) function
:func:`raster_plot` not to create a new figure (so that it can be placed
as a subplot of a larger figure). 
'''
# Constant input signal rate
signal = np.ones(len(ThalamicRate[0])) * v_0

# Compute total LFP
total_LFP = []
mean_LFP = []
for lfp in range(len(LFP_e.times)):
    total_LFP.append(np.sum(LFP_e[:, lfp]))
    mean_LFP.append(np.mean(LFP_e[:, lfp]))

# Convert LFP to millivolts
total_LFP = np.array(total_LFP) * 1e3
mean_LFP = np.array(mean_LFP) * 1e3

# Compute total DFP
total_DFP = []
mean_DFP = []
for dfp in range(len(DFP_e.times)):
    total_DFP.append(np.sum(DFP_e[:, dfp]))
    mean_DFP.append(np.mean(DFP_e[:, dfp]))

# Convert DFP to millivolts
total_DFP = np.array(total_DFP) * 1e3
mean_DFP = np.array(mean_DFP) * 1e3

# Compute mean membrane potential
mean_membrane = np.mean(Membrane.V, axis=0) * 1e3  # Convert from volts to millivolts

dt=0.0001 # 0.1ms = 0.0001 sec
#nextpow2=32768 # here 2000ms of length
nextpow2=4096 # only 250ms of length 


# Plot Thalamic Rate
plt.figure(figsize=(8, 6))
plt.plot(ThalamicRate.times / ms, ThalamicRate[0] / 1000, 'k', label='Thalamic Rate')
plt.plot(ThalamicRate.times / ms, signal / 1000, 'g', label='Constant Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Rate (kHz)')
plt.title('Thalamic Rate with Constant Signal')
plt.legend()
plt.savefig("ThalamicRate.eps", format='eps')
plt.show()

# Plot Excitatory Raster Plot
plt.figure(figsize=(8, 6))
spike_times = Ge_S.t / ms  # Assuming Ge_S contains spike times
neuron_indices = Ge_S.i
plt.eventplot(spike_times, colors='black')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Excitatory Raster Plot')
plt.savefig("Excitatory_RasterPlot.eps", format='eps')
plt.show()

# Plot Excitatory Rate
plt.figure(figsize=(8, 6))
plt.plot(rate_e.times / ms, rate_e.rate, 'b')
plt.xlabel('Time (ms)')
plt.ylabel('Rate (Hz)')
plt.title('Excitatory Population Rate')
plt.savefig("Excitatory_rate.eps", format='eps')
plt.show()

# Plot Inhibitory Rate
plt.figure(figsize=(8, 6))
plt.plot(rate_i.times / ms, rate_i.rate, 'r')
plt.xlabel('Time (ms)')
plt.ylabel('Rate (Hz)')
plt.title('Inhibitory Population Rate')
plt.savefig("Inhibitory_rate.eps", format='eps')
plt.show()

# Plot Mean Membrane Potential
plt.figure(figsize=(8, 6))
plt.plot(Membrane.times / ms, mean_membrane, 'g')
plt.xlabel('Time (ms)')
plt.ylabel('Potential (mV)')
plt.title('Mean Membrane Potential')
plt.savefig("mean_membrane.eps", format='eps')
plt.show()

# Plot Total LFP
plt.figure(figsize=(8, 6))
plt.plot(LFP_e.times / ms, total_LFP, 'm')
plt.xlabel('Time (ms)')
plt.ylabel('Total LFP')
plt.title('Total LFP')
plt.savefig("total_LFP.eps", format='eps')
plt.show()

# Plot Mean LFP
plt.figure(figsize=(8, 6))
plt.plot(LFP_e.times / ms, mean_LFP, 'm')
plt.xlabel('Time (ms)')
plt.ylabel('Mean LFP')
plt.title('Mean LFP')
plt.savefig("mean_LFP.eps", format='eps')
plt.show()

# Plot Total DFP
plt.figure(figsize=(8, 6))
plt.plot(DFP_e.times / ms, total_DFP, 'c')
plt.xlabel('Time (ms)')
plt.ylabel('Total DFP')
plt.title('Total DFP')
plt.savefig("total_DFP.eps", format='eps')
plt.show()

# Plot Mean DFP
plt.figure(figsize=(8, 6))
plt.plot(DFP_e.times / ms, mean_DFP, 'c')
plt.xlabel('Time (ms)')
plt.ylabel('Mean DFP')
plt.title('Mean DFP')
plt.savefig("mean_DFP.eps", format='eps')
plt.show()

# Plot Power Spectra
plt.figure(figsize=(8, 6))
plt.psd(total_LFP, NFFT=4096, Fs=1/dt)
plt.title('Power Spectra of Total LFP')
plt.savefig("PowerSpectra_total_LFP.eps", format='eps')
plt.show()

plt.figure(figsize=(8, 6))
plt.psd(mean_LFP, NFFT=4096, Fs=1/dt)
plt.title('Power Spectra of Mean LFP')
plt.savefig("PowerSpectra_mean_LFP.eps", format='eps')
plt.show()

plt.figure(figsize=(8, 6))
plt.psd(total_DFP, NFFT=4096, Fs=1/dt)
plt.title('Power Spectra of Total DFP')
plt.savefig("PowerSpectra_total_DFP.eps", format='eps')
plt.show()

plt.figure(figsize=(8, 6))
plt.psd(mean_DFP, NFFT=4096, Fs=1/dt)
plt.title('Power Spectra of Mean DFP')
plt.savefig("PowerSpectra_mean_DFP.eps", format='eps')
plt.show()

plt.figure(figsize=(8, 6))
plt.psd(mean_membrane, NFFT=4096, Fs=1/dt)
plt.title('Power Spectra of Mean Membrane Potential')
plt.savefig("PowerSpectra_mean_membrane.eps", format='eps')
plt.show()

##### --------------------------END --------------------------------