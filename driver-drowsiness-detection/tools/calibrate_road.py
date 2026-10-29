"""Check the road-camera calibration before trusting lane / car-position logic.

Draws the road trapezoid (ROAD_ROI in src/driver_assist.py) on a frame and shows the bird's-eye view that the
lane detector works on. Good calibration = the two lane lines are vertical and parallel in the right-hand image.

    python tools/calibrate_road.py --image road.png --out check.png           # from a saved frame
    python tools/calibrate_road.py --camera 1                                 # grab one frame, show it (any key closes)
    python tools/calibrate_road.py --image road.png --roi "500,470 780,470 1200,680 120,680"

--roi lists four x,y points in the 1280x720 frame: top-left top-right bottom-right bottom-left.
Copy the values you settle on into ROAD_ROI at the top of src/driver_assist.py. See docs/calibration.md.
"""
import argparse
import sys

import cv2
import numpy as np

# Must equal ROAD_ROI in src/driver_assist.py (tests/test_tools.py enforces it)
DEFAULT_ROI = "540,460 754,460 1190,670 160,670"
FRAME_SIZE = (1280, 720)


def parse_roi(text):
    points = []
    for chunk in text.split():
        x, y = chunk.split(",")
        points.append((int(x), int(y)))
    if len(points) != 4:
        raise ValueError(f"--roi needs exactly 4 points (top-left top-right bottom-right bottom-left), got {len(points)}")
    contour = np.int32(points)
    if (len(set(points)) != 4 or
            any(not (0 <= x < FRAME_SIZE[0] and 0 <= y < FRAME_SIZE[1]) for x, y in points) or
            not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 1):
        raise ValueError("--roi must be four distinct, convex, ordered points within 1280x720")
    return points


def render(frame, roi):
    """Return (frame with the ROI drawn, bird's-eye view), mirroring the app's perspective transform."""
    frame = cv2.resize(frame, FRAME_SIZE)
    top_left, top_right, bottom_right, bottom_left = roi
    src = np.float32([top_left, top_right, bottom_right, bottom_left])
    dst = np.float32([(bottom_left[0], 0), (bottom_right[0], 0), (bottom_right[0], bottom_right[1]), (bottom_left[0], bottom_left[1])])
    warped = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), FRAME_SIZE, flags=cv2.INTER_LINEAR)
    annotated = frame.copy()
    cv2.polylines(annotated, [np.int32(src)], True, (0, 255, 255), 3)
    for name, p in zip(("TL", "TR", "BR", "BL"), roi):
        cv2.circle(annotated, p, 6, (0, 0, 255), -1)
        cv2.putText(annotated, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return annotated, warped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="saved road-camera frame")
    source.add_argument("--camera", type=int, help="camera index to grab one frame from")
    parser.add_argument("--roi", default=DEFAULT_ROI, help=f'four points (default: "{DEFAULT_ROI}")')
    parser.add_argument("--out", help="write the side-by-side result here instead of opening a window")
    args = parser.parse_args(argv)

    try:
        roi = parse_roi(args.roi)
    except ValueError as exc:
        parser.error(str(exc))
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"cannot read image: {args.image}")
    else:
        cap = cv2.VideoCapture(args.camera)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"cannot read a frame from camera {args.camera}")

    annotated, warped = render(frame, roi)
    both = np.hstack([cv2.resize(annotated, (640, 360)), cv2.resize(warped, (640, 360))])
    if args.out:
        if not cv2.imwrite(args.out, both):
            sys.exit(f"cannot write image: {args.out}")
        print(f"wrote {args.out}")
    else:
        cv2.imshow("road ROI (left) and bird's-eye view (right) - any key closes", both)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
