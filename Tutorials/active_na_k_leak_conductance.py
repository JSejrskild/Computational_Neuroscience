"""
active_na_k_conductance.py

Simulates a simplified Hodgkin-Huxley-style model with Na+, K+, and leak channels.
Includes activation (m, n) and inactivation (h) gating variables and shows how
the membrane responds to varying input amplitudes.

Author: Matthew J. Crossley
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# === Constants ===
K_in = 140
K_out = 5
Na_in = 15
Na_out = 145
E_K = 61 * np.log10(K_out / K_in)
E_Na = 61 * np.log10(Na_out / Na_in)
E_L = -65
g_K_max = 10
g_Na_max = 15
g_L = 0.1

# === Gating kinetics ===
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10 + 1e-6))
def beta_m(V): return 4.0 * np.exp(-(V + 65) / 18)
def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10 + 1e-6))
def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

# === Simulation parameters ===
T = 300
dt = 0.1
t = np.arange(0, T, dt)
N = len(t)

pulse_amps = [0.5, 5.0, 10.0]
colors = cm.Reds(np.linspace(0.4, 0.9, len(pulse_amps)))
pulse_width = N // 6
start = N // 3
end = start + pulse_width

# === Storage ===
V_traces, I_K_traces, I_Na_traces, I_L_traces = [], [], [], []
g_K_traces, g_Na_traces = [], []
m_traces, h_traces, n_traces = [], [], []
I_ext_traces = []

for amp in pulse_amps:
    I_ext = np.zeros(N)
    I_ext[start:end] = amp

    V = np.zeros(N)
    V[0] = -65
    m = np.zeros(N)
    h = np.ones(N)
    n = np.zeros(N)
    g_K = np.zeros(N)
    g_Na = np.zeros(N)
    I_K = np.zeros(N)
    I_Na = np.zeros(N)
    I_L = np.zeros(N)

    m[0] = alpha_m(V[0]) / (alpha_m(V[0]) + beta_m(V[0]))
    h[0] = alpha_h(V[0]) / (alpha_h(V[0]) + beta_h(V[0]))
    n[0] = alpha_n(V[0]) / (alpha_n(V[0]) + beta_n(V[0]))

    g_K[0] = g_K_max * n[0]**4
    g_Na[0] = g_Na_max * m[0]**3 * h[0]

    I_K[0] = g_K[0] * (V[0] - E_K)
    I_Na[0] = g_Na[0] * (V[0] - E_Na)
    I_L[0] = g_L * (V[0] - E_L)

    for i in range(1, N):
        a_m, b_m = alpha_m(V[i-1]), beta_m(V[i-1])
        a_h, b_h = alpha_h(V[i-1]), beta_h(V[i-1])
        a_n, b_n = alpha_n(V[i-1]), beta_n(V[i-1])

        tau_m = 1 / (a_m + b_m)
        tau_h = 1 / (a_h + b_h)
        tau_n = 1 / (a_n + b_n)

        m_inf = a_m * tau_m
        h_inf = a_h * tau_h
        n_inf = a_n * tau_n

        m[i] = m[i-1] + (m_inf - m[i-1]) / tau_m * dt
        h[i] = h[i-1] + (h_inf - h[i-1]) / tau_h * dt
        n[i] = n[i-1] + (n_inf - n[i-1]) / tau_n * dt

        g_K[i] = g_K_max * n[i]**4
        g_Na[i] = g_Na_max * m[i]**3 * h[i]

        I_K[i] = g_K[i] * (V[i-1] - E_K)
        I_Na[i] = g_Na[i] * (V[i-1] - E_Na)
        I_L[i] = g_L * (V[i-1] - E_L)

        dVdt = - (I_K[i] + I_Na[i] + I_L[i]) + I_ext[i-1]
        V[i] = V[i-1] + dVdt * dt

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

for i, I in enumerate(I_ext_traces):
    ax[0].plot(t, I, label=f'{pulse_amps[i]} mV', color=colors[i])
ax[0].set_ylabel('Input')
ax[0].set_title('External Input Currents')

for i, V in enumerate(V_traces):
    ax[1].plot(t, V, color=colors[i])
ax[1].set_ylabel('Membrane V (mV)')
ax[1].set_title('Membrane Potential')

for i, I in enumerate(I_K_traces):
    ax[2].plot(t, I, color=colors[i])
ax[2].set_ylabel('K⁺ Current\n(+ out, - in)')

for i, I in enumerate(I_Na_traces):
    ax[3].plot(t, I, color=colors[i])
ax[3].set_ylabel('Na⁺ Current\n(+ out, - in)')

for i, I in enumerate(I_L_traces):
    ax[4].plot(t, I, color=colors[i])
ax[4].set_ylabel('Leak Current\n(+ out, - in)')

for i, g in enumerate(g_K_traces):
    ax[5].plot(t, g, color=colors[i])
ax[5].set_ylabel('g_K')

for i, g in enumerate(g_Na_traces):
    ax[6].plot(t, g, color=colors[i])
ax[6].set_ylabel('g_Na')

for i, m in enumerate(m_traces):
    ax[7].plot(t, m, color=colors[i])
ax[7].set_ylabel('Na⁺ m(t)')

for i, h in enumerate(h_traces):
    ax[8].plot(t, h, color=colors[i])
ax[8].set_ylabel('Na⁺ h(t)')

for i, n in enumerate(n_traces):
    ax[9].plot(t, n, color=colors[i])
ax[9].set_ylabel('K⁺ n(t)')
ax[9].set_xlabel('Time (ms)')

ax[0].legend(loc='upper right')
plt.tight_layout()
plt.show()

