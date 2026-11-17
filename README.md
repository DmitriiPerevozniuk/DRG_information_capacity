# DRG cross-excitation model

Minimal simulation code accompanying **Cross-excitation in dorsal root ganglia
redistributes coding capacity from stimulus intensity to stimulus location**,
by Dmitrii Perevozniuk, Pavel Musienko and Nikolay Koshev.

This release contains the model only: a one-dimensional population of leaky
integrate-and-fire neurons coupled by a decaying chemical agent. It intentionally
excludes plotting, parameter fitting, parameter sweeps, network comparisons,
IOU/mutual-information analysis and precomputed results. It is not a pipeline for
regenerating the paper's figures.

## Run

Python 3.11–3.13 is recommended. From this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python example.py
.venv/bin/python -m unittest -v
```

On Debian/Ubuntu, creating a virtual environment may require the matching
`python3-venv` package. The first simulation includes Numba compilation time.
The example performs one 400-neuron, 2000-ms simulation with dt = 0.05 ms;
allow roughly 0.6–1 GB of free RAM for arrays and working allocations.
Nothing is saved unless requested:

```sh
.venv/bin/python example.py --seed 1 --output simulation.npz
```

The optional output contains `time`, `voltage`, `agent`, `spikes`, `current`,
`noise`, and `seed`. Generated results and caches are ignored by Git.

## Python interface

```python
from model import LIF_with_CE

time, voltage, agent, spikes, current, noise = LIF_with_CE(
    N=400, T=2000, dt=0.05,
    neighbor_radius=3.5, CE_spread='equal',
    tau_ce=220, g_ce=0.22,
    neuron_interval=(70, 330), time_stim=(10, 2000),
    mode='weak_periodic', freq=50, prob=0.75,
    pulse_width_ms=2.5, adjust_power=True,
    background_noise=True, background_prob=0.05, background_power=10,
    DoD=0, seed=0, return_mask=True,
)
```

These are representative single-run settings, not a claim that every paper
experiment used identical stimulation or noise. The population size, duration,
time step and fitted CE constants follow the paper. Inputs are voltage-equivalent
drives in the original implementation, not calibrated physical currents.

- Time quantities are in milliseconds; `freq` is in Hz. Neuron intervals are
  zero-based, half-open `[start, end)`. Multiple intervals are supported, e.g.
  `neuron_interval=((70, 100), (200, 230))`.
- `voltage`, `agent`, `spikes`, `current` and `noise` have shape
  `(len(time), N)`. Voltage and agent histories are float32, spikes are boolean,
  and input/background arrays are float64. `return_mask=False` omits `noise`.
- `neighbor_radius` is the distance in neuron indices to **each** side; the line
  does not wrap. A fractional radius uses the same integer-distance cutoff.
  `CE_spread='gaussian'` uses sigma = radius / 3, truncated at the radius.
- `DoD` is the degree of disorder: the fraction of functional neuron indices
  whose physical positions are randomly permuted before constructing coupling.
- Resting potentials are uniform about `v_rest=-65` with `v_spread=10` (−75 to
  −55). Threshold offsets are uniform from 12.5 to 37.5, or fixed at 25 when
  `vary_thresh=False`. Reset is rest minus 10; refractory duration is `t_ref=2`.
- `mode` supports `random`, `periodic` and `weak_periodic`. In the last mode,
  `prob` is sampled independently for each neuron at **each pulse time step**,
  not once per pulse. Periodic mode ignores `prob`. Random mode is not gated
  by periodic pulses.
- `adjust_power=True` sets pulse amplitude to `750 / pulse_width_ms`, retaining
  the notebook's fixed amplitude × width normalization (75 × 10). It overrides
  `power`. Use `adjust_power=False` to apply the requested amplitude directly.
- Use the same integer `seed` to repeat a run, and different seeds for independent
  replicates. Exact results across different NumPy/Numba/platform versions are
  not guaranteed. The caller's global NumPy RNG is not modified.

## Numerical conventions retained from the source

The released kernel implements the **code**, not a reinterpretation of the
manuscript equations. With previous-step spikes `s`, its update is:

```text
C_deposit = C + W @ s
C_next    = C_deposit * (1 - dt / tau_ce)
V_next    = V + (-(V - V_rest) + I_ext + g_ce * C_next) * dt / tau_m
```

The voltage update applies only outside the refractory period. Crossing threshold
records a spike and resets voltage. The refractory counter is `int(t_ref / dt)`.
Agent decay continues during refractoriness. Histories at row `i` are recorded
after agent deposition but before decay and voltage integration; spikes at row
`i` record the subsequent update, so they are not simultaneous post-update states.

Important details for interpreting results:

1. **Self-feedback is retained.** Equal coupling includes `W[i, i] = 1`, including
   at radius zero. Thus radius zero does not remove all chemical feedback. Set
   **`g_ce=0`** to disable its voltage effect completely. Gaussian coupling at
   radius zero is a special all-zero matrix, as in the notebook.
2. **The printed voltage equation differs from the implementation.** The kernel
   scales both input and `g_ce * C` by `1 / tau_m`; the manuscript describes a
   `C / tau_ce` term. No attempt was made to reconcile or silently replace this.
3. Pulses are aligned to simulation time zero and clipped to the stimulation
   window; they do not restart at `time_stim[0]`. Period and pulse width are
   truncated to integer numbers of time steps.
4. Background is sampled per neuron per time step across the full simulation
   when the clipped stimulation window is nonempty. Active stimulus samples
   **replace**, rather than add to, background. The returned `noise` is the
   background **before replacement**, matching the notebook's `mask` convention.
   Consequently, `current - noise` is not the exact stimulus amplitude at samples
   where stimulus replaced nonzero background. An empty stimulation window
   returns zero input and zero noise, preserving the original early-return behavior.


See `requirements.txt` for runtime dependencies.

## License

Released under the [MIT License](LICENSE), permitting reuse, modification and
redistribution with the copyright and license notice retained.

Repository: https://github.com/DmitriiPerevozniuk/DRG_information_capacity

The final paper DOI/citation will be added when available.
