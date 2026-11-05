"""Small model checks; run with python -m unittest -v."""

import unittest

import numpy as np

from model import LIF_with_CE, connectivity, create_stimulation, run_lif_kernel


class ModelTests(unittest.TestCase):
    def small_run(self, **kwargs):
        parameters = dict(N=8, T=40.0, dt=0.1, neuron_interval=(2, 6),
                          time_stim=(0, 40), freq=100, pulse_width_ms=2.5,
                          neighbor_radius=1, return_mask=True)
        parameters.update(kwargs)
        return LIF_with_CE(**parameters)

    def test_shapes_types_and_initial_state(self):
        time, voltage, agent, spikes, current, noise = self.small_run()
        self.assertEqual(time.shape, (400,))
        for array in (voltage, agent, spikes, current, noise):
            self.assertEqual(array.shape, (400, 8))
            self.assertTrue(np.isfinite(array).all())
        self.assertEqual(voltage.dtype, np.float32)
        self.assertEqual(agent.dtype, np.float32)
        self.assertEqual(spikes.dtype, np.bool_)
        self.assertTrue(np.all(agent[0] == 0))
        self.assertTrue(np.all((-75 <= voltage[0]) & (voltage[0] <= -55)))
        self.assertGreater(spikes.sum(), 0)

    def test_seed_and_no_global_rng_side_effect(self):
        np.random.seed(73)
        state = np.random.get_state()
        first = self.small_run(DoD=0.75)
        after = np.random.get_state()
        for before, value in zip(state, after):
            np.testing.assert_array_equal(before, value)
        second = self.small_run(DoD=0.75)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        different = self.small_run(seed=1, DoD=0.75)
        self.assertFalse(np.array_equal(first[1][0], different[1][0]))
        self.assertFalse(np.array_equal(first[4], different[4]))

    def test_connectivity_conventions(self):
        np.testing.assert_array_equal(connectivity(5, 0), np.eye(5))
        np.testing.assert_array_equal(connectivity(5, 0, 'gaussian'), np.zeros((5, 5)))
        weights = connectivity(5, 1.5)
        self.assertEqual(weights[0, 4], 0)  # No wrap-around.
        self.assertEqual(weights[0, 1], 1)
        self.assertEqual(weights[0, 2], 0)
        gaussian = connectivity(5, 3, 'gaussian')
        self.assertEqual(gaussian[2, 2], 1)
        self.assertGreater(gaussian[2, 3], gaussian[2, 4])
        shuffled = connectivity(8, 1, DoD=1)
        np.testing.assert_array_equal(shuffled, shuffled.T)
        np.testing.assert_array_equal(np.sort(shuffled.sum(0)),
                                      np.sort(connectivity(8, 1).sum(0)))

    def test_periodic_phase_and_noise_overwrite(self):
        current, noise = create_stimulation(
            30, 4, 1.0, mode='periodic', neuron_interval=((1, 2), (3, 4)),
            time_interval_ms=(3, 25), power=75, freq=100, pulse_width_ms=2,
            background_noise=True, background_prob=1, background_power=10,
            return_mask=True)
        expected = np.full((30, 4), 10.0)
        for start in (10, 20):
            expected[start:start + 2, 1] = 75
            expected[start:start + 2, 3] = 75
        np.testing.assert_array_equal(current, expected)
        np.testing.assert_array_equal(noise, np.full((30, 4), 10.0))

    def test_empty_window_and_no_noise(self):
        for window in ((0, 20), (50, 60), (10, 10)):
            current, noise = create_stimulation(
                20, 4, 1, neuron_interval=(0, 4), time_interval_ms=window,
                background_noise=False, return_mask=True)
            self.assertEqual(current.shape, (20, 4))
            np.testing.assert_array_equal(noise, np.zeros((20, 4)))
            if window[0] >= 20 or window[0] == window[1]:
                self.assertFalse(current.any())

    def test_weak_periodic_probability_extremes(self):
        parameters = dict(num_steps=20, N=4, dt=1, neuron_interval=(0, 4),
                          time_interval_ms=(0, 20), background_noise=False,
                          freq=100, pulse_width_ms=2)
        periodic = create_stimulation(**parameters, mode='periodic')
        weak = create_stimulation(**parameters, mode='weak_periodic', prob=1)
        np.testing.assert_array_equal(periodic, weak)
        self.assertFalse(create_stimulation(**parameters, prob=0).any())
        self.assertFalse(create_stimulation(**parameters, mode='random', prob=0).any())

    def test_power_normalization_and_return_length(self):
        result = self.small_run(background_noise=False, prob=1)
        self.assertEqual(result[4].max(), 750 / 2.5)
        self.assertEqual(len(self.small_run(return_mask=False)), 5)

    def test_true_feedback_off(self):
        first = self.small_run(g_ce=0, neighbor_radius=0)
        second = self.small_run(g_ce=0, neighbor_radius=7)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[3], second[3])

    def test_kernel_against_scalar_euler_reference(self):
        steps, N, dt = 20, 2, 0.5
        weights = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        rest = np.array([-65, -60], dtype=np.float32)
        threshold = rest + 1
        reset = rest - 10
        current = np.full((steps, N), 100.0)
        v, c, previous = rest.copy(), np.zeros(N, np.float32), np.zeros(N, np.float32)
        voltage, agent = np.zeros((steps, N), np.float32), np.zeros((steps, N), np.float32)
        spikes = np.zeros((steps, N), bool)
        run_lif_kernel(steps, N, dt, 220.0, 0.22, 20.0, rest, threshold, reset,
                       weights, current, v, c, previous, voltage, agent, spikes, 2.0)
        ref_v, ref_c = rest.astype(float), np.zeros(N)
        ref_spikes, counters = np.zeros(N), np.zeros(N, int)
        for i in range(steps):
            ref_c += weights @ ref_spikes
            np.testing.assert_allclose(voltage[i], ref_v, atol=2e-5)
            np.testing.assert_allclose(agent[i], ref_c, atol=2e-5)
            ref_c *= 1 - dt / 220
            for j in range(N):
                if counters[j] > 0:
                    counters[j] -= 1
                    ref_v[j] = reset[j]
                    ref_spikes[j] = 0
                else:
                    ref_v[j] += (-(ref_v[j] - rest[j]) + current[i, j]
                                 + 0.22 * ref_c[j]) * dt / 20
                    ref_spikes[j] = ref_v[j] >= threshold[j]
                    if ref_spikes[j]:
                        ref_v[j] = reset[j]
                        counters[j] = int(2 / dt)
            np.testing.assert_array_equal(spikes[i], ref_spikes.astype(bool))

    def test_invalid_parameters(self):
        for parameters in (dict(dt=0), dict(tau_ce=0), dict(DoD=2),
                           dict(CE_spread='other'), dict(prob=-1), dict(freq=0),
                           dict(neuron_interval=(-1, 3)), dict(seed=-1),
                           dict(pulse_width_ms=0), dict(time_stim=(30, 10))):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                self.small_run(**parameters)


if __name__ == '__main__':
    unittest.main()
