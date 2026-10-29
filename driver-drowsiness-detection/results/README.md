# Training results

Ultralytics YOLOv8 training output for seven runs. Each folder keeps the run configuration (`args.yaml`), the
per-epoch metrics (`results.csv`) and, where they exist, `results.png` (loss/metric curves), `confusion_matrix.png`,
`PR_curve.png` and `F1_curve.png`. Weights and the dataset-derived image mosaics (`train_batch*.jpg`, `val_batch*.jpg`)
are intentionally not included.

Metrics below are the **last epoch** of each run on that run's own validation split. Different runs used different
datasets, so the numbers are not comparable across rows.

| Run | Start weights | Dataset (`data:`) | Epochs | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| train | `yolov8n.pt` | `drowsinessyolov8-1/data.yaml` | 100 | 0.992 | 0.995 | 0.995 | 0.785 |
| train2 | `best1.pt` | `data.yaml` | 100 | 0.901 | 0.913 | 0.935 | 0.645 |
| train3 | `best3.pt` | `facial-fatigue-1/data.yaml` | 100 | 0.961 | 0.973 | 0.987 | 0.696 |
| train4 | `yolov8n.pt` | `drowsinessyolov8-2/data.yaml` | 100 | 0.999 | 0.995 | 0.995 | 0.781 |
| train5 | `best_1.pt` | `data.yaml` | 100 | 0.910 | 0.922 | 0.923 | 0.620 |
| train6 | `best_2.pt` | `facial-fatigue-1/data.yaml` | 100 | 0.982 | 0.994 | 0.984 | 0.677 |
| train7 | `yolov8m-oiv7.pt` | `drowsinessyolov8-2/data.yaml` | 9 | 0.974 | 0.960 | 0.988 | 0.687 |

Notes
- Folder names are as they appeared in the original archive. The `name:` / `save_dir:` fields inside `args.yaml` reflect run numbering on the original machine and do not always match the folder name (e.g. `train5` was saved as `train14`).
- `train7` stopped after 9 epochs and produced no plots.
- The shipped checkpoint records `train14` and the generic `data.yaml`, consistent with the `train5` configuration. Its selected-checkpoint metrics differ from this final-epoch table; see `../docs/model-card.md`. Missing earlier weights and dataset identity prevent full training reproduction.
