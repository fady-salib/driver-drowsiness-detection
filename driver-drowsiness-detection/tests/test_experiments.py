"""Tests for the scripts in experiments/ using a fake YOLO and fake keyboard input."""
import contextlib
import importlib.util
import io
import os
import sys
import types
import unittest
from unittest import mock

import cv2

import fakes

EXPERIMENTS = os.path.join(fakes.REPO, "experiments")


class FakeResult:
    def __init__(self, class_ids):
        self.boxes = [types.SimpleNamespace(cls=[c]) for c in class_ids]


def load(script, frames, names=None, quit_after=None):
    """Import experiments/<script> with a fake YOLO that streams `frames` (lists of class ids)."""
    calls = {"kwargs": None, "destroyed": 0, "consumed": 0}
    names = names or {0: "awake", 1: "drowsy"}

    class YOLO:
        def __init__(self, path):
            self.path, self.names = path, names

        def __call__(self, **kwargs):
            calls["kwargs"] = kwargs
            assert kwargs.get("stream") is True, "must stream, otherwise Ultralytics buffers every frame in memory"
            for ids in frames:
                calls["consumed"] += 1
                yield FakeResult(ids)

    module = types.ModuleType("ultralytics")
    module.YOLO = YOLO
    keys = iter([-1] * (quit_after or 10 ** 6) + [ord("q")])
    spec = importlib.util.spec_from_file_location(script[:-3], os.path.join(EXPERIMENTS, script))
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"ultralytics": module}):
        spec.loader.exec_module(mod)
    patches = [mock.patch.object(cv2, "waitKey", lambda *_: next(keys)),
               mock.patch.object(cv2, "destroyAllWindows", lambda: calls.__setitem__("destroyed", calls["destroyed"] + 1))]
    return mod, calls, patches


def run_main(mod, patches, argv=None):
    out = io.StringIO()
    with contextlib.ExitStack() as stack, contextlib.redirect_stdout(out):
        for p in patches:
            stack.enter_context(p)
        mod.main(argv or [])
    return out.getvalue()


class FlagDemo(unittest.TestCase):
    def test_prints_only_on_change_and_uses_model_class_names(self):
        frames = [[0], [0], [1], [1], [0], []]
        mod, calls, patches = load("drowsiness_flag_demo.py", frames)
        out = run_main(mod, patches)
        self.assertEqual(out.count("drowsiness_flag"), 3)          # awake -> drowsy -> awake (no box = awake)
        self.assertIn("DROWSY", out)
        self.assertEqual(calls["destroyed"], 1)

    def test_flipped_class_order(self):
        mod, calls, patches = load("drowsiness_flag_demo.py", [[1], [0]], names={0: "drowsy", 1: "awake"})
        out = run_main(mod, patches)
        self.assertLess(out.index("awake"), out.index("DROWSY"))   # id 1 is awake here, id 0 drowsy

    def test_q_quits_even_when_no_face_is_detected(self):
        """Regression: the q check used to sit inside `if boxes:`, so it was ignored while nothing was detected."""
        mod, calls, patches = load("drowsiness_flag_demo.py", [[]] * 50, quit_after=3)
        run_main(mod, patches)
        self.assertEqual(calls["consumed"], 4)
        self.assertEqual(calls["destroyed"], 1)

    def test_model_without_drowsy_class_is_rejected(self):
        mod, calls, patches = load("drowsiness_flag_demo.py", [[0]], names={0: "cat"})
        with self.assertRaises(SystemExit):
            run_main(mod, patches)


class WebcamDemo(unittest.TestCase):
    def test_streams_and_quits_on_q(self):
        mod, calls, patches = load("webcam_demo.py", [[0]] * 50, quit_after=2)
        run_main(mod, patches, ["--source", "2", "--save"])
        self.assertEqual(calls["kwargs"]["source"], 2)
        self.assertTrue(calls["kwargs"]["save"])
        self.assertEqual(calls["consumed"], 3)

    def test_video_file_source_stays_a_string(self):
        mod, calls, patches = load("webcam_demo.py", [[0]], quit_after=0)
        run_main(mod, patches, ["--source", "clip.mp4"])
        self.assertEqual(calls["kwargs"]["source"], "clip.mp4")
        self.assertFalse(calls["kwargs"]["save"])


if __name__ == "__main__":
    unittest.main()
