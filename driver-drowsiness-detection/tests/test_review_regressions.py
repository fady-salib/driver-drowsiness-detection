"""Review regressions run with fake cameras and pins only."""
import types
import unittest
import numpy as np
import fakes
from fakes import MOTOR, BUZZER, SERVO, RIGHT_DUTY, values


class ReviewRegressions(unittest.TestCase):
    def test_import_does_not_touch_gpio(self):
        r = fakes.run_app(run_name="import_only")
        self.assertFalse(r.ns["GPIO_AVAILABLE"])
        self.assertEqual(r.events, [])

    def test_default_gate_never_energizes_motor_or_steers(self):
        r = fakes.run_app(n_frames=35, stub_lane=True, allow_model_actuation=False)
        self.assertIsNone(r.error)
        self.assertTrue(r.ns["GPIO_AVAILABLE"])
        self.assertNotIn(1, values(r.events, "out", MOTOR))
        self.assertEqual(values(r.events, "servo", SERVO), [])
        self.assertIn(1, values(r.events, "out", BUZZER))

    def test_no_driver_detection_is_not_awake(self):
        r = fakes.run_app(n_frames=2, driver_boxes=[])
        self.assertIsInstance(r.error, RuntimeError)
        self.assertIn("unavailable", str(r.error))
        self.assertNotIn(1, values(r.events, "out", MOTOR))

    def test_low_or_invalid_detection_confidence_is_unknown(self):
        ns = fakes.load_module()
        decide = ns["driver_state"]
        def box(cls, conf):
            return types.SimpleNamespace(cls=[cls], conf=[conf])
        for cls, conf in [(0, .2), (1, float("nan")), (0, 1.1), (99, .9), (float("nan"), .9)]:
            self.assertIsNone(decide([box(cls, conf)]))
        self.assertEqual(decide([box(0, .9), box(1, .8)]), 0)

    def test_new_car_blocks_turn_on_that_same_frame(self):
        # Determine first possible turn, then inject the obstacle on that exact frame.
        clear = fakes.run_app(stub_lane=True, n_frames=35)
        turn = next(e for e in clear.events if e.kind == "servo" and e.value == RIGHT_DUTY)
        frame = round((turn.t - 1000) / .25)
        blocked = fakes.run_app(stub_lane=True, n_frames=frame,
                               car_box=fakes.CAR_IN_RIGHT_LANE, car_from=frame)
        self.assertNotIn(RIGHT_DUTY, values(blocked.events, "servo", SERVO))

    def test_slow_inference_rejects_observation(self):
        r = fakes.run_app(n_frames=2, inference_delay=2)
        self.assertIsInstance(r.error, RuntimeError)
        self.assertIn("deadline", str(r.error))
        self.assertNotIn(1, values(r.events, "out", MOTOR))

    def test_shutdown_motor_off_attempt_survives_buzzer_failure(self):
        r = fakes.run_app(n_frames=2)
        shutdown = r.hooks["atexit"][0]
        live = shutdown.__globals__
        live["_shutdown_done"] = False
        gpio = live["GPIO"]
        original = gpio.output
        def output(pin, value):
            if pin == BUZZER:
                raise RuntimeError("broken buzzer output")
            original(pin, value)
        gpio.output = output
        shutdown()
        self.assertEqual(values(r.events, "out", MOTOR)[-1], 0)

    def test_degenerate_lane_fit_is_rejected(self):
        windows = fakes.load_module()["sliding_windows"]
        image = np.zeros((720, 1280), np.uint8)
        image[700, :] = 1
        left, right, *_ = windows(image)
        self.assertIsNone(left)
        self.assertIsNone(right)

    def test_invalid_distance_is_not_nearby(self):
        estimate = fakes.load_module()["estimate_distance"]
        for width in (0, -1, float("nan")):
            self.assertEqual(estimate(width, 0), float("inf"))
