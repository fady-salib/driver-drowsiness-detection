# Initial audit and remediation — 2026-10-29

Base: `d20b311` on `main`. Full source application, tests/fakes, both experiments,
calibration tool, all notebook code cells, dataset metadata, seven training CSVs
and configurations, READMEs, workflow and checkpoint metadata were inspected
before edits. Plot artifacts were inventoried; numerical findings use CSVs, not
visual estimates. See `file-inventory.md` for each original tracked path.

## Confirmed versus unverified

Confirmed: YOLOv8 object detection of awake/drowsy, two-camera loop, OpenCV lane
threshold/warp/sliding windows, stock YOLO car detection integration, timed
alerts and legacy motor/steering requests, GPIO polarity, RPM callback and
width-based distance estimate. The root README's EAR/MAR, MIT license badge,
edge-optimization/accident-prevention implications are unsupported. No heart-rate
sensor or seat vibration exists. Project period Aug 2024–Jun 2025 is owner-provided.
No actual Pi/camera/relay/sensor validation is documented by this review.

Baseline 40 tests passed after installing NumPy/OpenCV; the first attempt failed
to import tests because OpenCV was missing. Tests were not assumed passing from
the previous README. The nested `.github/workflows/tests.yml` is not discovered
by GitHub because workflows must reside at the repository root.

## Prioritized findings

| Priority | Finding | Change / residual limit |
|---|---|---|
| P1 | Importing helpers configured GPIO and could enable motor requests during ordinary execution | Explicit `initialize_gpio(--enable-gpio)`; motor and steering behind default-disabled source gate; import has no pin effects |
| P1 | No detections meant awake, clearing warnings/restarting motor requests | Separate confidence/label interpreter: unknown terminates with cleanup; drowsy takes priority |
| P1 | Steering reads previous-frame road/car state | Current-frame perception moved before decisions; new obstacle regression on exact first-turn frame |
| P1 | Shutdown used one try block: buzzer failure could skip motor-off | Motor-off first; each action attempted independently; immediate loop `finally` cleanup |
| P1 | Ambiguous industrial sensor voltage advice | Remove unverified direct wiring claims; require manufacturer-rated isolation/level interface; no physical validation |
| P2 | Wall-clock timers can jump; timestamp before inference; no processing-age check | Monotonic timing, decision timestamp after inference, elapsed processing budget; camera-buffer age still unknown, blocking calls cannot be interrupted |
| P2 | RPM reset races callback and divides by nominal 1 s despite delayed loops | Lock count snapshot/reset; divide by actual elapsed interval; callback records state without GPIO writes |
| P2 | Lane fit accepts insufficient y diversity/crossing lines | Require distinct rows and finite positive widths; geometry remains heuristic |
| P2 | Invalid car boxes/distance can corrupt estimates | Reject nonfinite/reversed boxes and invalid confidence/classes; invalid width returns infinity |
| P2 | Calibration tool accepts degenerate ROI and reports failed writes as success | Convex/in-bounds/distinct-point validation and output-write error |
| P2 | Workflow is nested/unpinned | Root workflow with explicit nested working directory and pinned lightweight test dependencies |
| P2 | Dataset YAML points one directory above the documented exports | Paths now reference sibling train/valid/test folders; images still absent |
| P2 | Notebook mask interpolation, probability IoU, video preprocessing/path/size errors | Nearest-neighbor masks; thresholded BinaryIoU; RGB/[0,1] video input matching training; path joins/output-size fixes; seed; mixed-extension loading; syntax only, not executed |
| P3 | Root README incomplete and contradicts nested implementation | Canonical source-backed root README; nested navigation; retain canonical CSV summaries and attribution |

## Important file roles

`src/driver_assist.py`: perception helpers, legacy response state and optional GPIO.
This remains a substantial single-file application; separation improves through
pure interpretation, explicit hardware initialization and perception-before-decision
ordering, without replacing its working camera pipeline. A full state-machine
module extraction is future work, not claimed complete.

`tests/fakes.py`: simulated cameras, model classes, monotonic clock and GPIO;
extended to inject unknown observations, delayed inference and appearing cars.
The legacy actuation override is test-only and does not enable actual pins.
`test_driver_assist.py`, `test_experiments.py`, `test_tools.py`: retained tests
updated for intentional safer behavior. `test_review_regressions.py`: new boundary
and safety regression scenarios.

`experiments/webcam_demo.py`: streamed driver-only display; preserved.
`experiments/drowsiness_flag_demo.py`: no-box state now unknown, not awake.
`tools/calibrate_road.py`: source-consistent perspective visualization with ROI
and write validation. `tools/model_report.py`: trusted-checkpoint inspection and
optional CPU smoke test. `models/best_2.pt`: unchanged; metadata/provenance in
`model-card.md` and `model-evidence.json`.

`notebooks/lane_segmentation_fcn.ipynb`: separate VGG16/FCN experiment, not called
by the application. TensorFlow/KITTI absent; no training/evaluation claim. Its
current pretrained VGG input is normalized [0,1] rather than VGG's canonical
preprocessing; retraining experiments should evaluate that choice. Source fixes
do not establish trained performance.

All seven `results/*/args.yaml` / `results.csv` pairs: run evidence; configs and
tables retained, generic data/earlier weights unresolved. PNG curves/confusion
matrices: existing artifacts, not re-generated or independent evaluation.
Four `data/*/data.yaml` files and eight export readmes: class order, paths and
attribution; no image-label or subject-level split verification possible.

## Hardware changes requiring separate review

Do not enable `ALLOW_MODEL_ACTUATION` on physical hardware merely because fake
tests pass. The existing simulated policy automatically requests motion again
after an awake observation; it has no manual re-arm, vehicle dynamics model,
validated distance estimate or maneuver planner. Keep the gate disabled. Test
alert/RPM outputs only with suitable interfaces on an isolated bench. Real
hardware needs electrical polarity/off-state checks, watchdog/interlocks,
calibrated pulse limits and independently reviewed manual-control behavior.

Shutdown is best effort, not a safety guarantee. Keeping an off-level output
configured cannot guarantee its electrical state after process exit, power loss,
SIGKILL or OS hang; use hardware defaults that leave the motor off when not driven.
