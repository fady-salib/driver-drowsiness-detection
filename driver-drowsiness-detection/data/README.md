# Datasets

This project used public [Roboflow Universe](https://universe.roboflow.com) datasets, all licensed
**CC BY 4.0**. The images are **not** redistributed here — only each dataset's `data.yaml` and Roboflow
readme (which carry the attribution). To reproduce training, export each dataset in *YOLOv8* format from the
URL below into `data/<name>/` (so you have `train/`, `valid/`, `test/` next to `data.yaml`).

| Folder | Source | Images | Classes (id order) |
|--------|--------|--------|--------------------|
| `Drowsiness-Detection-5` | [cairo-university-tnmct / drowsiness-detection-gxmok](https://universe.roboflow.com/cairo-university-tnmct/drowsiness-detection-gxmok) (v5) | 552 | `0 awake`, `1 drowsy` |
| `drowsinessyolov8-2` | [drowsinessdetection-c54gb / drowsinessyolov8](https://universe.roboflow.com/drowsinessdetection-c54gb/drowsinessyolov8) (v2) | 1,768 | `0 awake`, `1 drowsy` |
| `facial-fatigue-1` | [local-project / facial-fatigue](https://universe.roboflow.com/local-project/facial-fatigue) (v1) | 2,262 | **`0 drowsy`, `1 awake`** |
| `tired-detect-3` | [smoke-tcfs5 / tired-detect](https://universe.roboflow.com/smoke-tcfs5/tired-detect) (v3) | 3,203 (incl. augmented copies) | `closed_eye`, `closed_mouth`, `open_eye`, `open_mouth` — **4 classes, eye/mouth state, not awake/drowsy** |

`drowsinessyolov8-1` (v1 of the second project) was used by the first training run but was not in the archive.
No `args.yaml` references `tired-detect-3`, so it is not known which run (if any) used it.

Class order differs across datasets. The main application resolves IDs from model
names and rejects models without both awake/drowsy labels; it does not hard-code
the dataset order. The four-class eye/mouth dataset is not directly compatible.

Split paths in these YAML files refer to `train/images`, `valid/images` and
`test/images` next to each YAML. Export there and pass the YAML's absolute path to
training. Images and their split assignments were not available in this review;
metadata cannot verify subject separation or augmentation/near-duplicate leakage.
