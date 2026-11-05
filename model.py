"""Leaky integrate-and-fire DRG model with chemical cross-excitation.

Extracted from CE_optimization.ipynb; see README.md for provenance and
implementation conventions. No plotting or analysis dependencies are required.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def run_lif_kernel(num_steps, N, dt, tau_ce, g_ce, tau_m, v_rest, v_thresh, v_reset,
                   W, I_ext, v, c, spikes, v_history, c_history, spike_train, t_ref):
    """Original forward-Euler simulation kernel (updates arrays in place)."""
    decay_c = 1.0 - (dt / tau_ce)
    inv_tau_m_dt = dt / tau_m
    ref_steps = int(t_ref / dt)
    ref_counts = np.zeros(N, dtype=np.int32)

    for i in range(num_steps):
        # Deposit agent from the PREVIOUS step's spikes, including self-feedback.
        for source in range(N):
            if spikes[source] > 0.0:
                for target in range(N):
                    c[target] += W[target, source]

        for j in range(N):
            # Histories precede decay/integration; spike_train records this update.
            v_history[i, j] = v[j]
            c_history[i, j] = c[j]
            c[j] *= decay_c

            if ref_counts[j] > 0:
                ref_counts[j] -= 1
                v[j] = v_reset[j]
                spikes[j] = 0.0
            else:
                I_total = I_ext[i, j] + (g_ce * c[j])
                v[j] += (-(v[j] - v_rest[j]) + I_total) * inv_tau_m_dt
                if v[j] >= v_thresh[j]:
                    v[j] = v_reset[j]
                    spikes[j] = 1.0
                    ref_counts[j] = ref_steps
                else:
                    spikes[j] = 0.0

            spike_train[i, j] = spikes[j]


def connectivity(N, neighbor_radius, CE_spread='equal', DoD=0.0, seed=0):
    """Return W[target, source] on a non-wrapping 1D line.

    DoD permutes the physical positions of a fraction of functional indices.
    Equal coupling retains the notebook's diagonal, even at radius zero.
    Gaussian coupling at radius zero is the notebook's special all-zero case.
    """
    if N <= 0 or int(N) != N:
        raise ValueError('N must be a positive integer')
    if not np.isfinite(neighbor_radius) or neighbor_radius < 0:
        raise ValueError('neighbor_radius must be finite and nonnegative')
    if not np.isfinite(DoD) or not 0 <= DoD <= 1:
        raise ValueError('DoD must be between 0 and 1')
    if CE_spread not in ('equal', 'gaussian'):
        raise ValueError("CE_spread must be 'equal' or 'gaussian'")

    indices = np.arange(N, dtype=np.float32)
    distance = np.abs(indices[:, None] - indices)
    if CE_spread == 'equal':
        W = (distance <= neighbor_radius).astype(np.float32)
    elif neighbor_radius == 0:
        W = np.zeros((N, N), dtype=np.float32)
    else:
        sigma = neighbor_radius / 3.0
        W = (np.exp(-(distance**2) / (2 * sigma**2))
             * (distance <= neighbor_radius)).astype(np.float32)

    if DoD > 0:
        rng = np.random.RandomState(seed)
        permutation = np.arange(N)
        count = int(DoD * N)
        if count > 1:
            selected = rng.choice(N, count, replace=False)
            permutation[selected] = rng.permutation(selected)
        W = W[np.ix_(permutation, permutation)]
    return W


@njit(cache=True)
def _create_stimulation_jit(num_steps, N, dt, mode, intervals, time_interval_ms,
                            power, prob, freq, pulse_width_ms, background_noise,
                            background_prob, background_power, seed):
    # Serial JIT keeps the legacy RandomState draws reproducible.
    np.random.seed(seed)
    I_ext = np.zeros((num_steps, N))
    noise = np.zeros((num_steps, N))
    t_start = max(0, min(int(time_interval_ms[0] / dt), num_steps))
    t_end = max(0, min(int(time_interval_ms[1] / dt), num_steps))
    duration_steps = t_end - t_start
    if duration_steps <= 0:
        return I_ext, noise

    if background_noise:
        random_mask = np.random.random((num_steps, N)) < background_prob
        I_ext[:] = np.where(random_mask, background_power, 0)
        noise[:] = I_ext

    for i in range(intervals.shape[0]):
        n_start, n_end = intervals[i, 0], intervals[i, 1]
        n_neurons_target = n_end - n_start
        if n_neurons_target <= 0:
            continue
        target_slice = I_ext[t_start:t_end, n_start:n_end]
        if mode == 'random':
            random_mask = np.random.random((duration_steps, n_neurons_target)) < prob
            target_slice[:] = np.where(random_mask, power, target_slice)
        else:
            period_steps = int((1000.0 / freq) / dt)
            width_steps = int(pulse_width_ms / dt)
            # Pulses are anchored to time zero, NOT the stimulation-window start.
            pulse_mask = (np.arange(t_start, t_end) % period_steps) < width_steps
            combined = np.empty((duration_steps, n_neurons_target), dtype=np.bool_)
            if mode == 'periodic':
                for k in range(n_neurons_target):
                    combined[:, k] = pulse_mask
            else:
                random_mask = np.random.random((duration_steps, n_neurons_target)) < prob
                for k in range(n_neurons_target):
                    combined[:, k] = pulse_mask & random_mask[:, k]
            # Stimulus replaces, rather than adds to, background at active samples.
            target_slice[:] = np.where(combined, power, target_slice)
    return I_ext, noise


def create_stimulation(num_steps, N, dt, mode='weak_periodic',
                       neuron_interval=(70, 330), time_interval_ms=(10, 2000),
                       power=75.0, prob=0.75, freq=50.0, pulse_width_ms=2.5,
                       background_noise=True, background_prob=0.05,
                       background_power=10.0, return_mask=False, seed=0):
    """Generate input, optionally returning the pre-overwrite background current.

    Intervals are half-open [start, end); times are in ms, frequency in Hz.
    In weak_periodic mode, prob is applied per neuron PER TIME STEP of a pulse,
    not once per whole pulse. The optional 'mask' is current, not a boolean mask.
    """
    if int(num_steps) != num_steps or num_steps <= 0 or int(N) != N or N <= 0:
        raise ValueError('num_steps and N must be positive integers')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    if mode not in ('random', 'periodic', 'weak_periodic'):
        raise ValueError('unknown stimulation mode')
    for name, value in [('prob', prob), ('background_prob', background_prob)]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be between 0 and 1')
    if not np.isfinite(power) or not np.isfinite(background_power):
        raise ValueError('input powers must be finite')
    if mode != 'random':
        if not np.isfinite(freq) or freq <= 0 or (1000.0 / freq) < dt:
            raise ValueError('frequency must give a period of at least one time step')
        if not np.isfinite(pulse_width_ms) or pulse_width_ms < dt:
            raise ValueError('pulse width must be at least one time step')
    window = np.asarray(time_interval_ms, dtype=float)
    if window.shape != (2,) or not np.isfinite(window).all() or window[1] < window[0]:
        raise ValueError('time interval must contain finite, ordered start/end times')
    intervals = np.asarray(neuron_interval)
    if intervals.shape == (2,):
        intervals = intervals.reshape(1, 2)
    if (intervals.ndim != 2 or intervals.shape[1] != 2
            or not np.isfinite(intervals).all()
            or not np.equal(intervals, np.floor(intervals)).all()):
        raise ValueError('neuron_interval must be an integer pair or sequence of pairs')
    if (np.any(intervals < 0) or np.any(intervals > N)
            or np.any(intervals[:, 1] < intervals[:, 0])):
        raise ValueError('neuron intervals must satisfy 0 <= start <= end <= N')
    if seed is None or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
        raise ValueError('seed must be an integer between 0 and 2**32 - 1')

    current, noise = _create_stimulation_jit(
        int(num_steps), int(N), float(dt), mode, intervals.astype(np.int64),
        tuple(window), float(power), float(prob), float(freq), float(pulse_width_ms),
        background_noise, float(background_prob), float(background_power), int(seed))
    return (current, noise) if return_mask else current


def LIF_with_CE(N=400, T=2000.0, dt=0.05, neighbor_radius=3.5,
                CE_spread='equal', tau_ce=220.0, g_ce=0.22,
                neuron_interval=(70, 330), time_stim=(10, 2000), tau_m=20.0,
                v_rest=-65.0, t_ref=2.0, power=75.0, freq=50.0, prob=0.75,
                vary_thresh=True, v_spread=10.0, background_noise=True,
                background_prob=0.05, background_power=10.0, mode='weak_periodic',
                return_mask=False, pulse_width_ms=2.5, adjust_power=True,
                DoD=0.0, seed=0):
    """Run one simulation; return time, voltage, agent, spikes, input[, noise].

    All matrices have shape (time steps, neurons). Voltage and agent are float32,
    spikes bool, input/noise float64. Histories are recorded before the update.
    Set g_ce=0 to disable chemical feedback completely. Seed controls all random
    draws; use different seeds for independent replicates.
    """
    scalars = (T, dt, tau_ce, g_ce, tau_m, v_rest, t_ref, v_spread)
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError('model parameters must be finite')
    if T <= 0 or dt <= 0 or tau_m <= 0 or tau_ce <= 0:
        raise ValueError('T, dt, tau_m and tau_ce must be positive')
    if dt > min(tau_ce, tau_m):
        raise ValueError('dt must not exceed tau_ce or tau_m for forward Euler')
    if t_ref < 0 or v_spread < 0:
        raise ValueError('t_ref and v_spread must be nonnegative')
    if seed is None or int(seed) != seed or not 0 <= seed <= 2**32 - 1:
        raise ValueError('seed must be an integer between 0 and 2**32 - 1')
    W = connectivity(N, neighbor_radius, CE_spread, DoD, seed)
    time = np.arange(0, T, dt)
    num_steps = len(time)
    if adjust_power:
        if not np.isfinite(pulse_width_ms) or pulse_width_ms <= 0:
            raise ValueError('pulse_width_ms must be positive when adjusting power')
        # Original expression power / (power * width / 750) simplifies to this.
        power = 750.0 / pulse_width_ms

    # Local legacy RNG preserves the notebook's seed-0 population without
    # changing the caller's NumPy RNG. Stimulation has its own seeded JIT RNG.
    rng = np.random.RandomState(seed)
    v = (np.ones(N, dtype=np.float32) * v_rest
         + 2 * (rng.random(N).astype(np.float32) - 0.5) * v_spread)
    rest = v.copy()
    if vary_thresh:
        threshold = v + (0.5 + rng.random(N).astype(np.float32)) * 25.0
    else:
        threshold = v + 25.0
    reset = v - 10.0
    c = np.zeros(N, dtype=np.float32)
    previous_spikes = np.zeros(N, dtype=np.float32)
    voltage = np.zeros((num_steps, N), dtype=np.float32)
    agent = np.zeros((num_steps, N), dtype=np.float32)
    spikes = np.zeros((num_steps, N), dtype=np.bool_)
    current, noise = create_stimulation(
        num_steps, N, dt, mode=mode, neuron_interval=neuron_interval,
        time_interval_ms=time_stim, power=power, prob=prob, freq=freq,
        pulse_width_ms=pulse_width_ms, background_noise=background_noise,
        background_prob=background_prob, background_power=background_power,
        return_mask=True, seed=seed)
    run_lif_kernel(num_steps, N, dt, tau_ce, g_ce, tau_m, rest, threshold, reset,
                   W, current, v, c, previous_spikes, voltage, agent, spikes, t_ref)
    result = (time, voltage, agent, spikes, current)
    return result + (noise,) if return_mask else result
