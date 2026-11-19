"""Tests for src/driver_assist.py. Run from the repo root:  python -m unittest discover -s tests -v

Standard library + numpy + opencv only. The real application loop runs against fake cameras/models/GPIO
(see fakes.py), so there is no hardware, GPU or ultralytics dependency.
"""
import signal
import unittest

import numpy as np

import fakes
from fakes import BUZZER, LED, MOTOR, SERVO, CENTER_DUTY, RIGHT_DUTY, run_app, first_time, values


class ClassIds(unittest.TestCase):
    def setUp(self):
        self.resolve = fakes.load_module()["resolve_class_ids"]

    def test_default_order(self):
        self.assertEqual(self.resolve({0: "awake", 1: "drowsy"}), (0, 1))

    def test_flipped_order_and_list_and_case(self):
        self.assertEqual(self.resolve({0: "drowsy", 1: "awake"}), (1, 0))
        self.assertEqual(self.resolve(["Drowsy", " AWAKE "]), (1, 0))

    def test_missing_class_is_an_error(self):
        with self.assertRaises(ValueError):
            self.resolve({0: "car", 1: "drowsy"})

    def test_app_follows_the_models_class_order(self):
        # facial-fatigue-1 style model: drowsy=0, awake=1. An awake driver must never raise an alert.
        awake = run_app(names={0: "drowsy", 1: "awake"}, drowsy_from=10 ** 9)
        self.assertEqual((awake.ns["AWAKE_CLASS_ID"], awake.ns["DROWSY_CLASS_ID"]), (1, 0))
        self.assertNotIn(1, values(awake.events, "out", BUZZER))
        drowsy = run_app(names={0: "drowsy", 1: "awake"})
        self.assertIn(1, values(drowsy.events, "out", BUZZER))

    def test_model_without_the_classes_exits_with_error_and_registers_safe_shutdown(self):
        r = run_app(names={0: "cat"})
        self.assertIsInstance(r.error, SystemExit)
        self.assertEqual(r.error.code, 1)
        self.assertIn("awake", r.out)
        self.assertTrue(r.hooks["atexit"])


class Curvature(unittest.TestCase):
    def setUp(self):
        ns = fakes.load_module()
        self.radius, self.scale, self.windows = ns["curvature_radius_m"], ns["lane_scale"], ns["sliding_windows"]
        self.ns = ns

    def test_recovers_a_known_radius_in_metres(self):
        R, y0 = 250.0, 20.0                          # true radius (m) and evaluation point (m)
        xm, ym = 3.7 / 700, 30.0 / 720
        # lane line x_m = (y_m - y0)^2 / (2R), expressed as a pixel-space fit
        fit = (ym ** 2 / (2 * R * xm), -ym * y0 / (R * xm), 0.0)
        self.assertAlmostEqual(self.radius(fit, y0 / ym, xm, ym), R, places=6)

    def test_straight_line_is_infinite(self):
        self.assertEqual(self.radius((0.0, 0.1, 300.0), 719, 0.005, 0.04), float("inf"))

    def test_scale_uses_the_measured_lane_width(self):
        left, right = (0, 0, 300.0), (0, 0, 1000.0)          # 700 px wide at the bottom
        xm, ym = self.scale(left, right, 720)
        self.assertAlmostEqual(xm, self.ns["LANE_WIDTH_M"] / 700)
        self.assertAlmostEqual(ym, self.ns["LANE_LENGTH_M"] / 720)

    def test_degenerate_width_falls_back(self):
        xm, _ = self.scale((0, 0, 500.0), (0, 0, 500.0), 720)
        self.assertAlmostEqual(xm, self.ns["LANE_WIDTH_M"] / 700.0)

    def test_no_lane_pixels_gives_no_fit(self):
        left_fit, right_fit, *_ = self.windows(np.zeros((720, 1280), np.uint8))
        self.assertIsNone(left_fit)
        self.assertIsNone(right_fit)

    def test_estimate_distance(self):
        self.assertAlmostEqual(self.ns["estimate_distance"](126, 80), 10.0)
        self.assertEqual(self.ns["estimate_distance"](0, 0), float("inf"))


class LanePipeline(unittest.TestCase):
    def setUp(self):
        self.process = fakes.load_module()["process_single_frame_for_lanes_and_cars"]
        self.live = self.process.__globals__          # runpy returns a copy; the function updates the live dict

    def test_blank_frame_does_not_crash_and_flags_lane_lost(self):
        self.live["lane_detection_ok"] = True         # prove it is actively switched off
        frame = np.zeros((720, 1280, 3), np.uint8)
        out = self.process(frame, 0)
        self.assertEqual(out.shape, frame.shape)
        self.assertFalse(self.live["lane_detection_ok"])

    def test_synthetic_road_is_detected(self):
        self.process(fakes.yellow_road(), 0)
        self.assertTrue(self.live["lane_detection_ok"])


class AlertSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_app()
        cls.ev = cls.r.events

    def test_runs_cleanly_with_hardware_enabled(self):
        self.assertIsNone(self.r.error)
        self.assertTrue(self.r.ns["GPIO_AVAILABLE"])
        self.assertIn("Combined detection application finished.", self.r.out)

    def test_escalation_timing(self):
        buzzer = first_time(self.ev, "out", BUZZER, 1)
        led = first_time(self.ev, "out", LED, 1)
        motor_off = first_time(self.ev, "out", MOTOR, 0)
        turn = first_time(self.ev, "servo", SERVO, RIGHT_DUTY)
        self.assertTrue(3.0 < led - buzzer <= 3.6, led - buzzer)
        self.assertTrue(3.0 < motor_off - led <= 3.6, motor_off - led)
        self.assertTrue(1.9 <= turn - led <= 2.6, turn - led)

    def test_normal_exit_leaves_a_safe_state(self):
        self.assertEqual(values(self.ev, "out", BUZZER)[-1], 0)
        self.assertEqual(values(self.ev, "out", LED)[-1], 0)
        self.assertEqual(values(self.ev, "out", MOTOR)[-1], 0)          # active-HIGH relay: LOW = off
        self.assertEqual(values(self.ev, "servo", SERVO)[-1], CENTER_DUTY)
        self.assertTrue(values(self.ev, "servo_stop", SERVO))

    def test_motor_pin_is_not_released_by_cleanup(self):
        channels = [e.value for e in self.ev if e.kind == "cleanup"][-1]
        self.assertNotIn(MOTOR, channels)

    def test_recovery_resets_everything(self):
        r = run_app(drowsy_until=40, n_frames=60)
        ev = [e for e in r.events if e.t < 1000 + 0.25 * 58]            # before the final shutdown
        self.assertEqual(values(ev, "out", BUZZER)[-1], 0)
        self.assertEqual(values(ev, "out", LED)[-1], 0)
        self.assertEqual(values(ev, "out", MOTOR)[-1], 1)               # motor back on
        self.assertEqual(values(ev, "servo", SERVO)[-1], CENTER_DUTY)


class ServoSetupRegression(unittest.TestCase):
    def test_without_servo_setup_the_hardware_is_silently_disabled(self):
        """Documents the original bug: GPIO.PWM() before GPIO.setup() raised and disabled all GPIO control."""
        r = run_app(patch_source=lambda s: s.replace("    GPIO.setup(SERVO_PWM_PIN, GPIO.OUT)", "    pass"))
        self.assertIsInstance(r.error, RuntimeError)
        self.assertEqual(values(r.events, "out", MOTOR)[-1], 0)

    def test_with_servo_setup_it_works(self):
        self.assertTrue(run_app(n_frames=10).ns["GPIO_AVAILABLE"])


class SteeringSafeguards(unittest.TestCase):
    def right_turns(self, r):
        return values(r.events, "servo", SERVO).count(RIGHT_DUTY)

    def test_no_car_steers_right(self):
        r = run_app(stub_lane=True)
        self.assertGreater(self.right_turns(r), 0)

    def test_car_in_right_outer_lane_blocks_steering(self):
        """Regression: the flag used to be reset before it was read, so this safeguard never worked."""
        r = run_app(stub_lane=True, car_box=fakes.CAR_IN_RIGHT_LANE)
        self.assertEqual(self.right_turns(r), 0)
        self.assertTrue(r.ns["car_in_right_outer_lane"])
        self.assertIsNotNone(first_time(r.events, "out", MOTOR, 0))     # still cuts the motor

    def test_lost_lane_lines_means_no_blind_steering(self):
        blank = np.full((720, 1280, 3), 70, np.uint8)
        r = run_app(road_image=blank)
        self.assertIsNone(r.error)
        self.assertEqual(self.right_turns(r), 0)
        self.assertIsNotNone(first_time(r.events, "out", MOTOR, 0))


class FailSafe(unittest.TestCase):
    def test_crash_then_safe_shutdown(self):
        r = run_app(crash_at=25)
        self.assertIsInstance(r.error, RuntimeError)
        self.assertEqual(values(r.events, "out", MOTOR)[-1], 0)         # finally shuts down immediately
        self.assertEqual(values(r.events, "out", BUZZER)[-1], 0)
        shutdown = r.hooks["atexit"][0]
        shutdown()                                                      # what the interpreter does at exit
        self.assertEqual(values(r.events, "out", MOTOR)[-1], 0)
        self.assertEqual(values(r.events, "out", BUZZER)[-1], 0)
        self.assertEqual(values(r.events, "out", LED)[-1], 0)
        self.assertEqual(values(r.events, "servo", SERVO)[-1], CENTER_DUTY)
        n = len(r.events)
        shutdown()
        self.assertEqual(len(r.events), n, "safe_shutdown must be idempotent")

    def test_sigterm_triggers_the_exit_path(self):
        r = run_app(n_frames=5)
        with self.assertRaises(SystemExit):
            r.hooks["signal"][signal.SIGTERM](signal.SIGTERM, None)

    def test_active_low_relay_polarity(self):
        r = run_app(patch_source=lambda s: s.replace("MOTOR_ACTIVE_HIGH = True", "MOTOR_ACTIVE_HIGH = False"))
        setup = [e for e in r.events if e.kind == "setup" and e.pin == MOTOR][0]
        self.assertEqual(setup.value, 1)                                # starts OFF = HIGH
        motor = values(r.events, "out", MOTOR)
        self.assertEqual(motor[0], 0)                                   # awake -> motor ON = LOW
        self.assertIn(1, motor)                                         # cut-off = HIGH
        self.assertEqual(motor[-1], 1)                                  # safe state = HIGH


class CommandLine(unittest.TestCase):
    def test_camera_indices(self):
        r = run_app(n_frames=3, argv=["driver_assist.py", "--driver-cam", "3", "--road-cam", "4"])
        self.assertEqual(r.caps, [3, 4])

    def test_defaults(self):
        self.assertEqual(run_app(n_frames=3).caps, [0, 1])


if __name__ == "__main__":
    unittest.main()
