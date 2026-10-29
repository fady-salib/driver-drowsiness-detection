"""Inspect the trusted shipped checkpoint and optionally run a blank-image CPU check.

PyTorch checkpoints may execute pickle code: do not use untrusted weights.
This tool does not train, evaluate accuracy, open cameras or configure GPIO.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "models/best_2.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    model = YOLO(args.model)
    checkpoint = model.ckpt
    report = {
        "sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "bytes": args.model.stat().st_size,
        "ultralytics_checkpoint_version": checkpoint.get("version"),
        "checkpoint_date": checkpoint.get("date"),
        "task": model.task,
        "names": model.names,
        "parameters_before_fusion": sum(p.numel() for p in model.model.parameters()),
        "architecture": model.model.yaml,
        "train_args": checkpoint.get("train_args"),
        "checkpoint_metrics": checkpoint.get("train_metrics"),
    }
    if args.smoke:
        result = model(np.zeros((384, 640, 3), np.uint8), device="cpu", verbose=False)[0]
        report["smoke_inference"] = {"input_shape": [384, 640, 3], "boxes": len(result.boxes),
                                     "device": "cpu", "purpose": "execution check, not accuracy evaluation"}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
