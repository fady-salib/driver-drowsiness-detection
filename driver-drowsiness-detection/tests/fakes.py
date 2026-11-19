"""Fake cameras, YOLO models, clock and RPi.GPIO so the real application loop can run headlessly in tests.

The fake GPIO enforces the real library's rules (setup() before output()/PWM()), so wiring mistakes show up.
Nothing here needs a Raspberry Pi, a camera, a GPU or the ultralytics package.
"""
import collections
import contextlib
import io
import itertools
import os
import runpy
import signal
import sys
import tempfile
import time
import types
from unittest import mock

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(REPO, "src", "driver_assist.py")
BUZZER, LED, MOTOR, SERVO, RPM = 17, 27, 22, 18, 26      # BCM pins used by the app
CENTER_DUTY, RIGHT_DUTY = 7.5, 8.06                      # 90 deg and 100 deg servo duty cycles

Event = collections.namedtuple("Event", "t kind pin value")
Result = collections.namedtuple("Result", "ns events out error caps hooks")

# Controlled lane overlay used instead of the real lane pipeline when a test needs a car "in" a lane
STUB_LANE = '''
def process_single_frame_for_lanes_and_cars(input_frame, video_index_param):
    global road, started, lane_detection_ok
    road = np.zeros_like(input_frame)
    road[300:720, 400:800]  = (0, 255, 255)     # ego lane (yellow)
    road[300:720, 800:1100] = (0, 255, 0)       # right outer lane (green)
    started = 1
    lane_detection_ok = True
    return input_frame.copy()

if __name__ == "__main__":'''
CAR_IN_RIGHT_LANE = [850, 440, 1050, 640]


def yellow_road():
    """Synthetic road with two yellow lane markings (detected by the real thresholding pipeline)."""
    img = np.full((720, 1280, 3), 70, np.uint8)
    cv2.line(img, (300, 700), (560, 460), (0, 215, 255), 16)
    cv2.line(img, (1050, 700), (760, 460), (0, 215, 255), 16)
    return img


def make_gpio(events, state):
    G = types.ModuleType("RPi.GPIO")
    G.BCM, G.OUT, G.IN, G.HIGH, G.LOW, G.PUD_UP, G.FALLING = 11, 0, 1, 1, 0, 22, 32
    ready = set()

    def setup(pin, mode, initial=None, pull_up_down=None):
        ready.add(pin)
        events.append(Event(state["t"], "setup", pin, initial))

    def output(pin, value):
        if pin not in ready:
            raise RuntimeError("The GPIO channel has not been set up as an OUTPUT")
        events.append(Event(state["t"], "out", pin, value))

    class PWM:
        def __init__(self, pin, freq):
            if pin not in ready:
                raise RuntimeError("You must setup() the GPIO channel first")

        def start(self, duty):
            pass

        def stop(self):
            events.append(Event(state["t"], "servo_stop", SERVO, None))

        def ChangeDutyCycle(self, duty):
            events.append(Event(state["t"], "servo", SERVO, round(duty, 2)))

    G.setmode = lambda mode: None
    G.setup, G.output, G.PWM = setup, output, PWM
    G.add_event_detect = lambda *a, **k: None
    G.cleanup = lambda channels=None: events.append(Event(state["t"], "cleanup", None, channels))
    return G


def run_app(*, drowsy_from=5, drowsy_until=10 ** 9, n_frames=140, dt=0.25, car_box=None, argv=None,
            patch_source=None, stub_lane=False, crash_at=None, names=None, road_image=None,
            install_gpio=True, run_name="__main__", allow_model_actuation=True, driver_boxes=None, car_from=0,
            inference_delay=0.0):
    """Run src/driver_assist.py end-to-end against fakes. Returns Result(ns, events, out, error, caps, hooks)."""
    names = names or {0: "awake", 1: "drowsy"}
    drowsy_id = [i for i, n in names.items() if n == "drowsy"]
    awake_id = [i for i, n in names.items() if n == "awake"]
    events, state, caps = [], {"t": 1000.0, "frame": 0}, []
    hooks = {"atexit": [], "signal": {}}
    clock = itertools.count(1000.0, dt)

    class Box:
        def __init__(self, cls, xyxy=None, conf=0.9):
            self.cls, self.xyxy, self.conf = [cls], [xyxy or [0, 0, 1, 1]], [conf]

    class Res:
        def __init__(self, boxes, frame):
            self.boxes, self._frame = boxes, frame

        def plot(self):
            return self._frame.copy()

    class YOLO:
        def __init__(self, path):
            self.path = str(path)
            self.is_car_model = "yolov8n" in self.path
            self.names = {0: "car"} if self.is_car_model else names

        def __call__(self, frame, verbose=False):
            if self.is_car_model:
                return [Res([Box(0, car_box)] if car_box and state["frame"] >= car_from else [], frame)]
            state["t"] += inference_delay
            if crash_at is not None and state["frame"] >= crash_at:
                raise RuntimeError("simulated crash inside the model")
            drowsy = drowsy_from <= state["frame"] < drowsy_until
            boxes = [Box(drowsy_id[0])] if (drowsy and drowsy_id) else ([Box(awake_id[0])] if awake_id else [])
            if driver_boxes is not None:
                boxes = [Box(*spec) for spec in driver_boxes]
            return [Res(boxes, frame)]

    class Cap:
        def __init__(self, index):
            self.role = len(caps)            # 0 = driver camera, 1 = road camera (construction order)
            caps.append(index)

        def isOpened(self):
            return True

        def read(self):
            if self.role == 0:
                state["frame"] += 1
                state["t"] = 1000.0 + state["frame"] * dt
            if state["frame"] > n_frames:
                return False, None
            return True, (road_image if road_image is not None else yellow_road()).copy()

        def release(self):
            pass

    def fake_time():
        return state["t"]

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = YOLO
    modules = {"ultralytics": ultralytics}
    if install_gpio:
        pkg = types.ModuleType("RPi")
        pkg.GPIO = make_gpio(events, state)
        modules.update({"RPi": pkg, "RPi.GPIO": pkg.GPIO})

    with open(APP) as fh:
        src = fh.read()
    if stub_lane:
        assert 'if __name__ == "__main__":' in src
        src = src.replace('if __name__ == "__main__":', STUB_LANE, 1)
    if allow_model_actuation:
        src = src.replace("ALLOW_MODEL_ACTUATION = False", "ALLOW_MODEL_ACTUATION = True")
    if patch_source:
        src = patch_source(src)
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "driver_assist.py")
    with open(path, "w") as fh:
        fh.write(src)

    out, error, ns = io.StringIO(), None, {}
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(sys.modules, modules))
        for name, value in (("VideoCapture", Cap), ("imshow", lambda *a, **k: None),
                            ("waitKey", lambda *a, **k: -1), ("destroyAllWindows", lambda: None)):
            stack.enter_context(mock.patch.object(cv2, name, value))
        stack.enter_context(mock.patch.object(time, "monotonic", fake_time))
        stack.enter_context(mock.patch.object(time, "sleep", lambda s: None))
        stack.enter_context(mock.patch("atexit.register", lambda fn, *a, **k: hooks["atexit"].append(fn)))
        stack.enter_context(mock.patch("signal.signal", lambda sig, h: hooks["signal"].__setitem__(sig, h)))
        stack.enter_context(mock.patch.object(sys, "argv", ([*(argv or ["driver_assist.py"]), "--enable-gpio"] if install_gpio else (argv or ["driver_assist.py"]))))
        stack.enter_context(contextlib.redirect_stdout(out))
        try:
            ns = runpy.run_path(path, run_name=run_name)
        except BaseException as exc:         # SystemExit and crashes are results, not test failures
            error = exc
    return Result(ns, events, out.getvalue(), error, caps, hooks)


def load_module():
    """Import the app's functions without running its main block."""
    return run_app(run_name="driver_assist_import").ns


def first_time(events, kind, pin, value):
    for e in events:
        if e.kind == kind and e.pin == pin and e.value == value:
            return e.t
    return None


def values(events, kind, pin):
    return [e.value for e in events if e.kind == kind and e.pin == pin]
