"""
active_na_k_conductance.py

Simulates a simplified Hodgkin-Huxley-style model with Na+, K+, and leak channels.
Includes activation (m, n) and inactivation (h) gating variables and shows how
the membrane responds to varying input amplitudes.

Author: Matthew J. Crossley
Extreme comments: Johanne S. Rejsenhus 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# === Constants ===
# These constants can be found in litterature from biology and is where these sinmulations have the "true" values from

K_in = 140 # Pottasium concentration inside the cell
K_out = 5 # Pottasium concentration outside the cell
Na_in = 15 # Sodium concentration inside the cell
Na_out = 145 # Sodium concentration outside the cell
E_K = 61 * np.log10(K_out / K_in) # Equilibrium potential for K+
E_Na = 61 * np.log10(Na_out / Na_in) # Equilibrium potential for Na+
E_L = -65 # Equilibrium potential for leak current - constant
g_K_max = 10 # Maximum K⁺ 
g_Na_max = 15 # Maximum Na⁺ 
g_L = 0.1 # Leak conductance


# === Gating kinetics ===
# Alphas and betas are the rate constants for the opening and closing of the ion channels
# Alphas activates the channels and betas inactivates them 

def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10 + 1e-6)) # rate of opening for Na+ channels after depolarization
def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18) # rate of closing for Na+ channels when repolarizing
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20) # rate of inactivation for Na+ channels despite depolarization
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10)) # rate of reactivation for Na+ channels - relative refractory period
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10 + 1e-6)) # rate of opening for K+ channels after depolarization -> in order to repolarize
def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80) # Rate of closing for K+ channels after repolarisation -> This is a bit slow, hence hyperpolarization

# === Simulation parameters ===
T = 300 # Total time to simulate (ms)
dt = 0.1 # Time step (ms)
t = np.arange(0, T, dt) # Time array -> 0 to 300 ms with 0.1 ms intervals
N = len(t) # Number of time points

# Is this the currents that are injected into the cell?
pulse_amps = [0.5, 5.0, 10.0] 
colors = cm.Reds(np.linspace(0.4, 0.9, len(pulse_amps))) 
pulse_width = N // 6 
start = N // 3
end = start + pulse_width

# === Storage ===
V_traces, I_K_traces, I_Na_traces, I_L_traces = [], [], [], [] # These are the traces of voltage and the three currents
g_K_traces, g_Na_traces = [], [] # Conductace of K+ and Na+
# Conductance is how easy the ions can flow through the channels/membrane
m_traces, h_traces, n_traces = [], [], [] # Gating variables for Na+ and K+ channels
I_ext_traces = [] # External input currents

for amp in pulse_amps:
    I_ext = np.zeros(N)  # External input is a  vector of zeros with the length of N
    I_ext[start:end] = amp # The external input is set to the amplitude of the pulse

# Here we initialize the variables that we need to keep track of as vectors of zeros
    V = np.zeros(N)
    V[0] = -65 # Starting membrane potential 
    m = np.zeros(N)
    h = np.ones(N)
    n = np.zeros(N)
    g_K = np.zeros(N) 
    g_Na = np.zeros(N)
    I_K = np.zeros(N)
    I_Na = np.zeros(N)
    I_L = np.zeros(N)

    m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0])) # Initial value of m
    h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0])) # Initial value of h
    n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0])) # Initial value of n

# Conductances
    g_K[0] = g_K_max * n[0]**4  # Initial value of K+ conductance
    g_Na[0] = g_Na_max * m[0]**3 * h[0] # Initial value of Na+ conductance
# Maybe ask Matthew about the "activation gates (3/4)" 
# and where they are expressed in the code or where that initial assumption comes from

# Currents
    I_K[0] = g_K[0] * (V[0] - E_K) # Initial value of K+ current
    I_Na[0] = g_Na[0] * (V[0] - E_Na) # Initial value of Na+ current
    I_L[0] = g_L * (V[0] - E_L)  # Initial value of leak current

# Now we are ready to simulate the model - Yay
    for i in range(1, N):
        # Update gating variables, so they follow the time steps 
        # Using the last value of the gating variable to calculate the new value
        a_m, b_m = alpha_m(V[i-1]), beta_m(V[i-1]) 
        a_h, b_h = alpha_h(V[i-1]), beta_h(V[i-1])
        a_n, b_n = alpha_n(V[i-1]), beta_n(V[i-1])

        tau_m = 1 / (a_m + b_m)
        tau_h = 1 / (a_h + b_h)
        tau_n = 1 / (a_n + b_n)

# Steady state values of the gating variables
        m_inf = a_m * tau_m # Probability of the Na+ channel being open at a given time
        h_inf = a_h * tau_h # Probability of the Na+ channel being inactivated at a given time
        n_inf = a_n * tau_n # Probability of the K+ channel being open at a given time

# Updating the gating variables
        m[i] = m[i-1] + (m_inf - m[i-1]) / tau_m * dt 
        h[i] = h[i-1] + (h_inf - h[i-1]) / tau_h * dt
        n[i] = n[i-1] + (n_inf - n[i-1]) / tau_n * dt

        g_K[i] = g_K_max * n[i]**4 # Here we again have the something to the power of 4 - maybe ask Matthew about this
        g_Na[i] = g_Na_max * m[i]**3 * h[i]

# Currents flowing through the channels given by the membrane potential V from the previous time step and the equilibrium potentials
        I_K[i] = g_K[i] * (V[i-1] - E_K) 
        I_Na[i] = g_Na[i] * (V[i-1] - E_Na)
        I_L[i] = g_L * (V[i-1] - E_L)

# Update membrane potential 
        dVdt = - (I_K[i] + I_Na[i] + I_L[i]) + I_ext[i-1] # The differential equation for the membrane potential
        V[i] = V[i-1] + dVdt * dt # Update the membrane potential

# Store the traces - this is the data that we want to plot :)
    V_traces.append(V) 
    I_K_traces.append(I_K)
    I_Na_traces.append(I_Na)
    I_L_traces.append(I_L)
    g_K_traces.append(g_K)
    g_Na_traces.append(g_Na)
    m_traces.append(m)
    h_traces.append(h)
    n_traces.append(n)
    I_ext_traces.append(I_ext)

# === Plotting ===
fig, ax = plt.subplots(10, 1, figsize=(10, 14), sharex=True)

# Plotting the external input currents
for i, I in enumerate(I_ext_traces):
    ax[0].plot(t, I, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[0].set_ylabel('Input')
ax[0].set_title('External Input Currents')

# Plotting the membrane potentials V
for i, V in enumerate(V_traces):
    ax[1].plot(t, V, color=colors[i])
ax[1].set_ylabel('Membrane V (mV)')
ax[1].set_title('Membrane Potential')

# Plotting the K+ currents
for i, I in enumerate(I_K_traces):
    ax[2].plot(t, I, color=colors[i])
ax[2].set_ylabel('K⁺ Current\n(+ out, - in)')

# Plotting the Na+ currents
for i, I in enumerate(I_Na_traces):
    ax[3].plot(t, I, color=colors[i])
ax[3].set_ylabel('Na⁺ Current\n(+ out, - in)')

# Plotting the leak currents
for i, I in enumerate(I_L_traces):
    ax[4].plot(t, I, color=colors[i])
ax[4].set_ylabel('Leak Current\n(+ out, - in)')

# Plotting the K+ conductance - how easy the ions can flow through the channels/membrane
for i, g in enumerate(g_K_traces):
    ax[5].plot(t, g, color=colors[i])
ax[5].set_ylabel('g_K')

# Plotting the Na+ conductance - how easy the ions can flow through the channels/membrane
for i, g in enumerate(g_Na_traces):
    ax[6].plot(t, g, color=colors[i])
ax[6].set_ylabel('g_Na')

# Plotting the gating variables - The probability of Na being open
for i, m in enumerate(m_traces):
    ax[7].plot(t, m, color=colors[i])
ax[7].set_ylabel('Na⁺ m(t)')

# Plotting the gating variables - The probability of Na being inactivated
for i, h in enumerate(h_traces):
    ax[8].plot(t, h, color=colors[i])
ax[8].set_ylabel('Na⁺ h(t)')

# Plotting the gating variables - The probability of K being open
for i, n in enumerate(n_traces):
    ax[9].plot(t, n, color=colors[i])
ax[9].set_ylabel('K⁺ n(t)')
ax[9].set_xlabel('Time (ms)')

ax[0].legend(loc='upper right')
plt.tight_layout()
plt.show()

