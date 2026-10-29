# Shipped model evidence — reviewed 2026-10-29

File: `models/best_2.pt`, 6,255,075 bytes; SHA-256
`b3c3c32f7abd722355bdab867163a1f5c93d64e2f49b93cbaf9489e95fc89697`.
The checkpoint was first inspected symbolically without invoking pickled model
classes, then loaded with the matching Ultralytics version for CPU execution.
Only trusted weights should be loaded; arbitrary PyTorch pickles can run code.

## Architecture and semantics

Stored task: `detect`, `ultralytics.nn.tasks.DetectionModel`. Architecture:
YOLOv8 nano-scale (depth multiplier 0.33, width multiplier 0.25), Conv/C2f backbone,
SPPF, feature concatenation/upsampling and three-scale Detect head. It returns
bounding boxes and per-box classes/confidences, not an image-classification
probability vector. The main application uses `{0: awake, 1: drowsy}` stored in
the checkpoint, resolves names dynamically and requires both labels.

Stored version: Ultralytics **8.3.39**. Checkpoint date:
`2025-02-26T16:23:09.914265`. See `model-evidence.json` for the complete
architecture, pre-fusion parameter count, training arguments and stored metrics.
No weight modifications or retraining occurred.

## Training evidence and limits

Metadata records `best_1.pt` as the starting model, `data.yaml`, run `train14`,
100 requested epochs, batch 16, image size 640, seed 0 and deterministic training.
This is consistent with the folder labeled `results/train5`, whose `args.yaml`
also records `train14`. Some arguments differ (checkpoint workers=0 versus
config workers=8; checkpoint warmup bias LR=0 versus config 0.1), so the exported
configuration is not a complete exact execution record. Earlier weights and
the actual generic dataset/split are missing.

| Evidence source | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| Selected checkpoint's stored validation metrics | 0.87693 | 0.95025 | 0.95565 | 0.65555 |
| `train5/results.csv`, final epoch 100 | 0.91046 | 0.92244 | 0.92303 | 0.61968 |

These are saved training/validation artifacts, not newly reproduced metrics.
The checkpoint is stripped (`epoch=-1`), so its selected epoch is not directly
identified by that field. Do not present the final row as a direct measurement
of the shipped weights. Seven CSVs were independently read; their last-epoch
values agree with `results/README.md`. Runs use different data and are not a
controlled architecture comparison. `train7` has only nine recorded epochs.

Dataset attribution cards list 552, 1,768, 2,262 and 3,203 exported images,
including augmentations where stated. Images are absent. The exact `train5`
dataset, class prevalence, subject separation, duplicate/augmentation leakage,
and test partition cannot be confirmed. No independent driver/lighting/camera
generalization evaluation, confidence calibration, false-alert rate or per-person
analysis is available. EAR/MAR, heart sensing and vibration are absent from code.

## Execution check

With Python 3.12, Ultralytics 8.3.39, PyTorch 2.5.1+cpu, TorchVision 0.20.1+cpu,
NumPy 2.2.6 and OpenCV headless 4.12.0.88, the model loaded and processed a
384×640 black RGB/BGR-equivalent image on CPU, returning zero boxes. This is an
execution check only; it provides no accuracy, latency benchmark or road-camera
model validation. No driver images or cameras were used.

Reproduce with `python tools/model_report.py --smoke --output model-report.json`.
The road camera uses a separate stock COCO model; its weights and a smoke test
are not included in this review. Third-party Ultralytics/dataset license terms
apply; this repository has no project license file.
