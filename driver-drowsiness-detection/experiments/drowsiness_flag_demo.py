"""Prints the drowsiness flag the main application uses: 1 = awake, 0 = drowsy, None = unknown (printed when it changes).

    python experiments/drowsiness_flag_demo.py [--source 0] [--model models/best_2.pt]

The drowsy class id is read from the model, so models trained with a different class order work too.
Click the video window and press q to quit.
"""
import argparse
import multiprocessing
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "best_2.pt"


def drowsy_class_id(names):
    ids = {str(name).strip().lower(): int(i) for i, name in names.items()}
    if "drowsy" not in ids:
        raise SystemExit(f"model has no 'drowsy' class (classes: {sorted(ids)})")
    return ids["drowsy"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index or video path (default: 0)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="weights file (default: models/best_2.pt)")
    args = parser.parse_args(argv)
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)
    drowsy_id = drowsy_class_id(model.names)
    awake_ids = {int(i) for i, name in model.names.items() if str(name).strip().lower() == "awake"}
    last_flag = object()
    for frame_results in model(source=source, show=True, stream=True, verbose=False):
        drowsy = any(int(box.cls[0]) == drowsy_id for box in frame_results.boxes)
        awake = any(int(box.cls[0]) in awake_ids for box in frame_results.boxes)
        flag = 0 if drowsy else (1 if awake else None)
        if flag != last_flag:
            print(f"drowsiness_flag = {flag}  ({'DROWSY' if drowsy else ('awake' if awake else 'unknown')})")
            last_flag = flag
        if cv2.waitKey(1) & 0xFF == ord("q"):      # checked on every frame, not only when a face is detected
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
