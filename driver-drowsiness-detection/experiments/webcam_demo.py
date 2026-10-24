"""Quick visual check of the drowsiness model on a webcam or a video file.

    python experiments/webcam_demo.py                      # webcam 0
    python experiments/webcam_demo.py --source clip.mp4    # video file
    python experiments/webcam_demo.py --save               # also write annotated output to runs/detect/

Click the video window and press q to quit.
"""
import argparse
import multiprocessing
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "best_2.pt"


def parse_source(text):
    return int(text) if text.isdigit() else text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index or video path (default: 0)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="weights file (default: models/best_2.pt)")
    parser.add_argument("--save", action="store_true", help="save the annotated output")
    args = parser.parse_args(argv)

    model = YOLO(args.model)
    # stream=True yields one result at a time; without it Ultralytics keeps every frame in memory
    for _ in model(source=parse_source(args.source), show=True, save=args.save, stream=True, verbose=False):
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
