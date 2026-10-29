# Calibrating the road camera

The lane and car-position logic in `src/driver_assist.py` depends on a few numbers that describe *your* camera
mount and *your* road. The defaults are placeholders from the original build — re-measure them for a new setup.
All of them are constants in the **Configuration** block at the top of `src/driver_assist.py`.

## 1. Road region — `ROAD_ROI`

The four corners (in the 1280×720 road frame) of the road area that is warped to a bird's-eye view.

1. Save a frame from the road camera while the vehicle sits straight on a straight piece of road with two lane lines visible:
   `python tools/calibrate_road.py --camera 1 --out check.png` (or use any saved frame with `--image`).
2. Pick the corners: **top** points on the two lane lines a little below the horizon, **bottom** points on the same lines near the bottom of the image.
   Order: top-left, top-right, bottom-right, bottom-left.
3. Check them: `python tools/calibrate_road.py --image frame.png --roi "x,y x,y x,y x,y" --out check.png`.
   In the right-hand (bird's-eye) image the two lane lines must be **vertical and parallel**. If they lean inwards or
   outwards, move the top points apart or together until they don't.
4. Copy the final numbers into `ROAD_ROI`.

## 2. Metres per pixel — `LANE_WIDTH_M`, `LANE_LENGTH_M`

Only used for the "Curvature radius" read-out, not for any decision.

- `LANE_WIDTH_M`: real distance between the two lane lines (3.7 m on many roads; measure your model road).
- `LANE_LENGTH_M`: real length of road covered by the bird's-eye view, i.e. the ground distance between the top and bottom edge of `ROAD_ROI`.

The metres-per-pixel across the lane is measured from the detected lines every frame, so only these two numbers are needed.

## 3. Car distance — `FOCAL_LENGTH_PIXELS`, `AVERAGE_CAR_WIDTH_METERS`

Distance is estimated as `AVERAGE_CAR_WIDTH_METERS × FOCAL_LENGTH_PIXELS / bounding-box width`. To calibrate:

1. Park a car (or your model car) at a **known distance** `D` metres straight ahead and note its real width `W`.
2. Read its detected bounding-box width `w` in pixels from the app's output.
3. Set `AVERAGE_CAR_WIDTH_METERS = W` and `FOCAL_LENGTH_PIXELS = w × D / W`.

`CAR_MAX_DISTANCE_M` is how close a car must be to count as "in" a lane.

## 4. Lane-line detection

Lines are found with an HLS saturation threshold plus a gradient threshold (`thresholding_pipeline`). If your
lines are not found (the video shows **"Lane lines not found"**), check lighting/contrast first and lower
`s_thresh` or `MIN_FIT_PIXELS`. While lane lines are not found the app **will not steer** (it still sounds the
alarm and cuts the motor).

The colour ranges in `driver_assist.py` (`purple_*`, `yellow_*`, `green_*`) match the overlay colours the app draws
itself, so they do **not** need to be tuned per scene.

## Review constraints

Physical motor/steering actuation is disabled by default. Lane/car decisions are
visualized and tested against fake pins only. The current-frame check is not a
validated maneuver planner. The width-based distance assumes a known visible
vehicle width; occlusion, orientation, bounding-box errors and unseen objects
invalidate it. Camera buffer age is unknown. Do not interpret an empty right-lane
flag as proof that a lane is safe.
