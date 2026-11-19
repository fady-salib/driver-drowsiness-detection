# Verification — 2026-10-29

Base `d20b311` (`main`). Review branch: `portfolio/driver-monitoring-audit`.

## Executed checks

- First baseline attempt: tests could not import because OpenCV was absent.
  After installing NumPy 2.2.6 / OpenCV headless 4.12.0.88 in a dedicated Python
  3.12 environment, **40 baseline tests passed in 33.438 seconds**.
- Final `python -m unittest discover -s tests -v`: **51 tests passed in 31.126
  seconds**. This includes retained tests, updated expectations for intentional
  safety changes, and eleven new regression methods. The expected printed
  broken-buzzer shutdown warning is part of a fault-injection test, not a silent
  test failure.
- `python -m compileall -q src tools tests experiments`: passed.
- Every notebook code cell parsed with `ast.parse`: passed. The notebook was
  **not executed or trained**; TensorFlow and KITTI images are absent.
- `tools/model_report.py --smoke`: shipped custom driver weights loaded using
  Ultralytics 8.3.39 / PyTorch 2.5.1+cpu / TorchVision 0.20.1+cpu; one 384×640
  blank-image CPU inference returned zero boxes. Checkpoint SHA-256/metadata are
  recorded in `model-evidence.json`. No detection accuracy or latency claim follows.
- All seven CSVs were read and final-epoch metrics checked against the saved
  results table. Saved selected-checkpoint metrics were separately extracted;
  no training, calibration, validation dataset inference or test-set evaluation
  was performed.

During model tooling setup the environment used the headless OpenCV substitute
for Ultralytics' GUI OpenCV dependency; do not interpret this as validation of a
desktop display install. A Fontconfig cache warning appeared during initial
import; checkpoint loading and inference still completed. The test environment
and application dependency declarations serve different GUI/hardware purposes.

## Scope of simulation

Tests use fake frames/models, monotonic time and GPIO. Legacy steering and motor
policy is exercised only after a **test-only source override** of the disabled
actuation gate; new default-gate tests verify no motor-on or steering command.
No Pi, camera capture, actual GPIO, industrial sensor, relay, servo, motor,
body sensor, or vehicle was connected. GPIO timing, relay electrical polarity,
shutdown pin retention and physical interlocks remain unverified.

The separate COCO road model was not downloaded or run. No held-out driver data
is included. The stored dataset metadata and curves cannot prove subject-level
independence or robust real-world performance. All hardware-affecting changes
need separate review on an isolated model platform with actuation still disabled
until that review is complete. No remote GitHub CI success is claimed here.

Publication is blocked by current GitHub write permissions (HTTP 403). A local
commit, patch, bundle and draft PR description are delivered separately; the
default branch remains unchanged.
