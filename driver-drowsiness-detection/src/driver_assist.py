from ultralytics import YOLO
import multiprocessing
import cv2
import numpy as np
import math # Retained for potential future use or if hidden dependencies exist
import time
import os
import sys
import atexit
import signal
import argparse
from pathlib import Path

# Trained weights live in <repo>/models, independent of the working directory
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ======================= Configuration — edit for your build =======================
# Motor relay polarity: True = GPIO HIGH energises the motor (active-HIGH relay); False = active-LOW relay module.
MOTOR_ACTIVE_HIGH = True

# Road-camera calibration. These MUST be re-measured for every camera mount / model road:
#  * ROAD_ROI: the road trapezoid (pixels, in the 1280x720 road frame) that is warped to a bird's-eye view
#  * LANE_WIDTH_M / LANE_LENGTH_M: real width of the lane and real length of road covered by the bird's-eye view
#    (they turn pixel measurements into metres for the curvature read-out)
ROAD_ROI = {"top_left": [540, 460], "top_right": [754, 460], "bottom_right": [1190, 670], "bottom_left": [160, 670]}
LANE_WIDTH_M = 3.7
LANE_LENGTH_M = 30.0
MIN_FIT_PIXELS = 150          # fewer lane pixels than this per line -> "lane not found" instead of a bogus fit

# Car distance estimate: distance = AVERAGE_CAR_WIDTH_METERS * FOCAL_LENGTH_PIXELS / bbox_width_px
AVERAGE_CAR_WIDTH_METERS = 1.8
FOCAL_LENGTH_PIXELS = 700     # camera-specific; calibrate against a car at a known distance
CAR_MAX_DISTANCE_M = 15.0     # cars further away than this are ignored
# ====================================================================================

# --- Raspberry Pi Specific Imports (Uncomment and configure if controlling physical GPIO) ---
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    GPIO.setmode(GPIO.BCM) # Use Broadcom SOC channel numbers (GPIO numbers, not physical pin numbers)

    # Define GPIO pins for actual hardware control (EXAMPLE PINS - ADJUST AS NEEDED FOR YOUR WIRING)
    # Refer to Raspberry Pi GPIO pinout diagrams for your specific model (e.g., Pi 4)
    # Use 'pinout' command in Raspberry Pi terminal for a visual guide.
    #
    # Pinout Examples (BCM numbering):
    # GPIO17 (Physical Pin 11): Can be used for Buzzer
    # GPIO27 (Physical Pin 13): Can be used for LED
    # GPIO22 (Physical Pin 15): Can be used for Motor Control via Relay
    # GPIO18 (Physical Pin 12): Supports Hardware PWM, ideal for Servo
    # GPIO26 (Physical Pin 37): Good for a digital input sensor like Autonics PR18-8DN

    BUZZER_PIN = 17 # Example GPIO pin for buzzer (e.g., connected to a transistor/relay driving a buzzer)
    LED_PIN = 27    # Example GPIO pin for LED
    MOTOR_CONTROL_PIN = 22 # Example GPIO pin for motor control (e.g., connected to a relay module's IN pin)
    SERVO_PWM_PIN = 18 # Example GPIO pin for servo motor (connect to signal pin of servo)
    RPM_SENSOR_PIN = 26 # GPIO pin for the Autonics PR18-8DN sensor (black wire)

    # --- Autonics PR18-8DN Sensor Wiring to Raspberry Pi ---
    # This sensor is NPN-NO (Normally Open).
    # - NPN: Output connects to Ground when an object is detected.
    # - NO (Normally Open): Output is open (not connected to GND) when no object is detected.
    #
    # Wiring Diagram:
    # --------------------------------------------------------------------------------------
    # Sensor Wire Color | Sensor Function | Raspberry Pi Connection (BCM Pin, Physical Pin)
    # --------------------------------------------------------------------------------------
    # Brown             | +V (Power)      | 3.3V or 5V (Physical Pin 1 or 2/4)
    #                   |                 | (Connect to 5V if sensor is 5V-rated, otherwise 3.3V)
    # Blue              | 0V (Ground)     | GND (Physical Pin 6/9/14 etc.)
    # Black             | Output Signal   | GPIO26 (Physical Pin 37)
    # --------------------------------------------------------------------------------------
    # The Pi's internal pull-up resistor (GPIO.PUD_UP) pulls GPIO26 HIGH.
    # When an object is detected, the sensor's black wire pulls GPIO26 LOW, triggering the interrupt (FALLING edge).

    # --- Relay Module Wiring to Raspberry Pi (for Motor Control) ---
    # This assumes a common low-level trigger relay module (active-LOW for trigger).
    # If your relay triggers high, adjust the logic (e.g., GPIO.HIGH to activate, LOW to deactivate).
    #
    # Wiring Diagram (Example for a 5V 1-channel Relay Module with Optocoupler):
    # ------------------------------------------------------------------------------------------
    # Relay Module Pin | Function       | Raspberry Pi Connection (BCM Pin, Physical Pin)
    # ------------------------------------------------------------------------------------------
    # VCC (or +)       | Power Supply   | 5V (Physical Pin 2 or 4) - For relay's internal circuit
    # GND (or -)       | Ground         | GND (Physical Pin 6/9/14 etc.)
    # IN (or DATA)     | Control Signal | GPIO22 (Physical Pin 15) - This is the `MOTOR_CONTROL_PIN`
    # ------------------------------------------------------------------------------------------
    # Relay Output (for Motor):
    # - Connect your motor's positive wire to "NO" (Normally Open) on the relay.
    # - Connect the other motor wire to your motor's power source (e.g., a separate battery).
    # - Connect "COM" (Common) on the relay to the positive terminal of your motor's power source.
    #   (OR, if controlling power to motor from a supply, COM to supply +, NO to motor +)
    # (The configuration for controlling the motor will depend on your motor type and driver.)

    # Setup GPIO pins as outputs
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.setup(MOTOR_CONTROL_PIN, GPIO.OUT, initial=(GPIO.LOW if MOTOR_ACTIVE_HIGH else GPIO.HIGH)) # Relay control pin, starts OFF
    GPIO.setup(SERVO_PWM_PIN, GPIO.OUT)     # Required before GPIO.PWM(), otherwise RuntimeError disables all hardware control

    # Setup PWM for servo on the specified pin
    # Frequency (Hz) for most hobby servos is 50 Hz.
    servo = GPIO.PWM(SERVO_PWM_PIN, 50) 
    servo.start(0) # Start PWM with 0% duty cycle (servo at neutral position or off)

    # Setup RPM sensor pin as input with internal pull-up resistor
    GPIO.setup(RPM_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    # Add event detection (interrupt) for the RPM sensor
    # GPIO.FALLING means trigger when voltage goes from HIGH to LOW (sensor detects object)
    # bouncetime helps prevent multiple triggers from a single event (e.g., sensor flutter)
    SENSOR_BOUNCETIME_MS = 20 # milliseconds for debounce
    GPIO.add_event_detect(RPM_SENSOR_PIN, GPIO.FALLING, callback=lambda x: rpm_count_rotation(), bouncetime=SENSOR_BOUNCETIME_MS)

    print("RPi.GPIO imported and configured. Hardware control enabled.")
    print(f"Buzzer connected to GPIO {BUZZER_PIN}")
    print(f"LED connected to GPIO {LED_PIN}")
    print(f"Motor control (via Relay) connected to GPIO {MOTOR_CONTROL_PIN}")
    print(f"Servo (Steering) connected to GPIO {SERVO_PWM_PIN}")
    print(f"RPM Sensor connected to GPIO {RPM_SENSOR_PIN}")

except ImportError:
    GPIO_AVAILABLE = False
    print("RPi.GPIO not available. Hardware control simulated.")
    print("To enable hardware control, install RPi.GPIO (pip install RPi.GPIO) and run on a Raspberry Pi.")
except RuntimeError:
    GPIO_AVAILABLE = False
    try:
        GPIO.cleanup()          # release anything that was configured before the failure
    except Exception:
        pass
    print("Error initializing RPi.GPIO. Are you running on a Raspberry Pi with root privileges? Hardware control simulated.")
    print("Try running the script with 'sudo python3 your_script_name.py'")


def set_motor(on):
    """Switch the motor relay on/off, honouring MOTOR_ACTIVE_HIGH. No-op without GPIO."""
    if GPIO_AVAILABLE:
        GPIO.output(MOTOR_CONTROL_PIN, GPIO.HIGH if (bool(on) == MOTOR_ACTIVE_HIGH) else GPIO.LOW)


_shutdown_done = False

def safe_shutdown():
    """Leave the hardware in a safe state however the program ends (normal exit, crash, Ctrl-C, SIGTERM):
    buzzer and LED off, steering centred, motor relay OFF. Idempotent.
    The motor pin is deliberately NOT passed to GPIO.cleanup(): cleanup would turn it back into a floating
    input, whereas an un-cleaned output keeps driving the relay at its OFF level after the process ends."""
    global _shutdown_done
    if _shutdown_done or not GPIO_AVAILABLE:
        return
    _shutdown_done = True
    try:
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.output(LED_PIN, GPIO.LOW)
        set_motor(False)
        servo.ChangeDutyCycle(90 / 18.0 + 2.5)     # centre the steering
        time.sleep(0.3)                             # give the servo time to get there
        servo.stop()
    except Exception as e:
        print(f"Warning: could not set every output to its safe state: {e}")
    finally:
        GPIO.cleanup([BUZZER_PIN, LED_PIN, SERVO_PWM_PIN, RPM_SENSOR_PIN])


# Default video index: 0 for generic webcam setup, 1 for specific video params
video_index = 0 # This is now just a setting for lane detection parameters

# --- Global Variables for Lane Detection ---
# These are shared between the main loop and the lane processing function
road = np.zeros((720, 1280, 3), dtype=np.uint8) # Lane overlay image
started = 0 # Flag to indicate if lane detection has started (road variable is populated)
lane_detection_ok = False # True only while the latest frame produced usable lane lines
car_in_left_outer_lane = False # Flag for car in left outer lane
car_in_ego_lane = False        # Flag for car in ego lane
car_in_right_outer_lane = False # Flag for car in right outer lane

# --- Drowsiness Detection Class IDs ---
# Defaults match models/best_2.pt; they are re-read from the loaded model at start-up.
AWAKE_CLASS_ID = 0
DROWSY_CLASS_ID = 1

def resolve_class_ids(names):
    """Return (awake_id, drowsy_id) from a YOLO `names` mapping or list, matching 'awake'/'drowsy' case-insensitively.
    Datasets disagree on class order (facial-fatigue-1 is drowsy=0, awake=1), so the order must never be assumed."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    by_name = {str(n).strip().lower(): int(i) for i, n in items}
    missing = [n for n in ("awake", "drowsy") if n not in by_name]
    if missing:
        raise ValueError(f"model classes {sorted(by_name)} must include 'awake' and 'drowsy' (missing: {missing})")
    return by_name["awake"], by_name["drowsy"]

# --- Global variables for buzzer flag, LED flag, and drowsiness timers ---
buzzer_flag = False # Controls continuous buzzer for drowsiness
led_flag = False    # New flag for the LED, activated after buzzer
drowsy_start_time = None
buzzer_on_time = None

# --- New Global variable for Steering Angle and its associated timers ---
steering_angle = 180 # Default steering angle (straight)
led_active_for_steering_time = None # To track when LED is on for triggering steering angle change (2 seconds delay)
steering_angle_210_start_time = None # To track when steering_angle changed to 210 (2 seconds duration)
steering_angle_180_cooldown_start_time = None # To track when steering_angle returned to 180 for cooldown (3 seconds minimum)

# --- New Global variable for Motor Flag and its associated timer ---
motor_flag = 1 # 1 indicates motor is active/on, 0 indicates motor is inactive/off
led_on_time = None # To store the timestamp when the LED flag first became True for motor delay

# --- Global variables for RPM detection ---
rpm_rotations = 0 # Volatile equivalent: incremented by interrupt
rpm_last_update_time = time.time() # Initialize with current time for first interval
current_rpm = 0.0 # Stores the calculated RPM
RPM_UPDATE_INTERVAL_MS = 1000 # Update RPM every 1000 milliseconds (1 second)

# --- RPM Momentary Buzzer Control ---
rpm_buzz_active = False # Flag to indicate if momentary RPM buzz is ongoing
rpm_buzz_start_time = 0 # Timestamp when momentary RPM buzz started
RPM_BUZZ_DURATION_MS = 50 # Duration for momentary RPM beep

# --- Wheel Speed Calculation Variables ---
WHEEL_DIAMETER_MM = 300 # Diameter of the wheel in millimeters
WHEEL_RADIUS_METERS = (WHEEL_DIAMETER_MM / 1000.0) / 2 # Radius in meters (0.15m for 300mm diameter)
wheel_speed_mps = 0.0 # Linear speed of the wheel in meters per second

# Interrupt Service Routine (ISR) for RPM sensor - called by RPi.GPIO event detect
def rpm_count_rotation():
    """
    Callback function for the RPM sensor GPIO event.
    Increments rotation count and initiates a momentary buzzer beep if allowed.
    """
    global rpm_rotations, rpm_buzz_active, rpm_buzz_start_time
    rpm_rotations += 1
    # Trigger a momentary beep ONLY if the main drowsiness buzzer is not active
    # and a previous RPM beep is not currently active
    if GPIO_AVAILABLE and not buzzer_flag and not rpm_buzz_active:
        GPIO.output(BUZZER_PIN, GPIO.HIGH) # Turn buzzer ON momentarily for RPM
        rpm_buzz_active = True
        rpm_buzz_start_time = time.time() # Start timer for this specific beep

# --- Helper functions for Lane Detection ---

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon formed from `vertices`.
    The rest of the image is set to black.
    """
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        ignore_mask_color = (255,) * img.shape[2]
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def thresholding_pipeline(img, s_thresh=(90, 255), sxy_thresh=(20, 100)):
    """
    Pipeline for combining color and gradient thresholding to identify lane pixels.
    """
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32) 
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / max(np.max(abs_sobelx), 1e-6))  # guard: uniform frame -> max 0

    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sxy_thresh[0]) & (scaled_sobelx <= sxy_thresh[1])] = 1
    grad_thresh = sxbinary

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    combined_binary = np.zeros_like(grad_thresh)
    combined_binary[(s_binary == 1) | (grad_thresh == 1)] = 1 
    return combined_binary

def perspective_transform(img, src_mask, dst_mask):
    """
    Applies perspective transformation to get a bird's-eye view of the road.
    """
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_mask)
    dst = np.float32(dst_mask)
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_img, M

def lane_scale(left_fit, right_fit, height_px):
    """(xm_per_pix, ym_per_pix) for the bird's-eye view. The lane width measured at the bottom of the image
    is taken to be LANE_WIDTH_M, and the image height to be LANE_LENGTH_M."""
    y = height_px - 1
    width_px = np.polyval(right_fit, y) - np.polyval(left_fit, y)
    xm_per_pix = LANE_WIDTH_M / width_px if width_px > 10 else LANE_WIDTH_M / 700.0
    return xm_per_pix, LANE_LENGTH_M / float(height_px)

def curvature_radius_m(fit, y_eval_px, xm_per_pix, ym_per_pix):
    """Radius of curvature in metres of a lane line x = A*y^2 + B*y + C fitted in pixel space.
    Changing units y_m = ym*y, x_m = xm*x turns the fit into A*xm/ym^2, B*xm/ym. Infinite for a straight line."""
    A, B, _ = fit
    a_m = A * xm_per_pix / (ym_per_pix ** 2)
    b_m = B * xm_per_pix / ym_per_pix
    if a_m == 0:
        return float("inf")
    y_m = y_eval_px * ym_per_pix
    return (1 + (2 * a_m * y_m + b_m) ** 2) ** 1.5 / abs(2 * a_m)

def sliding_windows(binary_warped, nwindows=9):
    """
    Implements sliding windows algorithm to find and fit a polynomial to lane lines.
    """
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):, :], axis=0)
    midpoint = np.int32(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int32(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) < MIN_FIT_PIXELS or len(rightx) < MIN_FIT_PIXELS:
        # night / glare / no markings: np.polyfit would raise or return nonsense
        return None, None, lefty, leftx, righty, rightx
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit, lefty, leftx, righty, rightx

def project_lanelines(binary_warped, orig_img_shape, left_fit, right_fit, Minv_local):
    """
    Projects lane line polygons back to the original image and draws them.
    """
    color_warp = np.zeros((binary_warped.shape[0], binary_warped.shape[1], 3), dtype=np.uint8)
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if len(left_fitx) > 0 and len(right_fitx) > 0:
        valid_widths = right_fitx - left_fitx
        valid_widths = valid_widths[valid_widths > 0]
        lane_width_pixels = np.mean(valid_widths) if len(valid_widths) > 0 else 300 
    else:
        lane_width_pixels = 300

    outer_left_fitx = left_fitx - lane_width_pixels
    outer_right_fitx = right_fitx + lane_width_pixels

    PURPLE = (128, 0, 128)  
    YELLOW = (0, 255, 255)  
    GREEN = (0, 255, 0)     
    RED_LINE = (0, 0, 255)  
    BLUE_LINE = (255, 0, 0) 
    LINE_THICKNESS = 5

    if len(outer_left_fitx) > 0 and len(left_fitx) > 0:
        pts_outer_left_fill = np.array([np.transpose(np.vstack([outer_left_fitx, ploty]))])
        pts_left_ego_fill = np.array([np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
        if pts_outer_left_fill.size > 0 and pts_left_ego_fill.size > 0:
            poly_pts_outer_left_combined = np.concatenate((pts_outer_left_fill, pts_left_ego_fill), axis=1)
            cv2.fillPoly(color_warp, [np.int32(poly_pts_outer_left_combined)], PURPLE)

    if len(left_fitx) > 0 and len(right_fitx) > 0:
        pts_left_ego_fill = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right_ego_fill = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        if pts_left_ego_fill.size > 0 and pts_right_ego_fill.size > 0:
            poly_pts_ego_combined = np.concatenate((pts_left_ego_fill, pts_right_ego_fill), axis=1)
            cv2.fillPoly(color_warp, [np.int32(poly_pts_ego_combined)], YELLOW)

    if len(right_fitx) > 0 and len(outer_right_fitx) > 0:
        pts_right_ego_fill = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        pts_outer_right_fill = np.array([np.flipud(np.transpose(np.vstack([outer_right_fitx, ploty])))])
        if pts_right_ego_fill.size > 0 and pts_outer_right_fill.size > 0:
            poly_pts_outer_right_combined = np.concatenate((pts_right_ego_fill, pts_outer_right_fill), axis=1)
            cv2.fillPoly(color_warp, [np.int32(poly_pts_outer_right_combined)], GREEN)

    warped_inv_overlay = cv2.warpPerspective(color_warp, Minv_local, orig_img_shape[:2][::-1], flags=cv2.INTER_LINEAR)
    
    if len(ploty) > 1:
        pts_left_line_warped = np.array([np.vstack([left_fitx, ploty]).T]).astype(np.float32)
        pts_right_line_warped = np.array([np.vstack([right_fitx, ploty]).T]).astype(np.float32)

        ego_left_line_pts_orig = cv2.perspectiveTransform(pts_left_line_warped, Minv_local)[0].astype(np.int32)
        ego_right_line_pts_orig = cv2.perspectiveTransform(pts_right_line_warped, Minv_local)[0].astype(np.int32)

        for i in range(1, len(ego_left_line_pts_orig)):
            if (0 <= ego_left_line_pts_orig[i-1][0] < orig_img_shape[1] and 
                0 <= ego_left_line_pts_orig[i-1][1] < orig_img_shape[0] and 
                0 <= ego_left_line_pts_orig[i][0] < orig_img_shape[1] and 
                0 <= ego_left_line_pts_orig[i][1] < orig_img_shape[0]):
                cv2.line(warped_inv_overlay, tuple(ego_left_line_pts_orig[i-1]), tuple(ego_left_line_pts_orig[i]), RED_LINE, LINE_THICKNESS)
            
            if (0 <= ego_right_line_pts_orig[i-1][0] < orig_img_shape[1] and 
                0 <= ego_right_line_pts_orig[i-1][1] < orig_img_shape[0] and 
                0 <= ego_right_line_pts_orig[i][0] < orig_img_shape[1] and 
                0 <= ego_right_line_pts_orig[i][1] < orig_img_shape[0]):
                cv2.line(warped_inv_overlay, tuple(ego_right_line_pts_orig[i-1]), tuple(ego_right_line_pts_orig[i]), BLUE_LINE, LINE_THICKNESS)
    return warped_inv_overlay

def process_single_frame_for_lanes_and_cars(input_frame, video_index_param):
    """
    Processes a single frame for lane detection and returns the annotated frame.
    This function replaces the previous 'pipeline' function's core logic.
    """
    global road, started, lane_detection_ok

    image = input_frame.copy() 
    
    height, width, _ = image.shape 

    # Note: video_index_param is passed but will now be used to select between different
    # pre-defined sets of perspective points if needed. For live cameras, these points
    # might need to be consistent regardless of video_index, or dynamically set.
    # Assuming video_index_param 0 for a generic webcam setup, if it's the road camera.
    if video_index_param == 0: # This refers to the specific settings for the road camera
        top_left, top_right = ROAD_ROI["top_left"], ROAD_ROI["top_right"]
        bottom_right, bottom_left = ROAD_ROI["bottom_right"], ROAD_ROI["bottom_left"]
    else: # Example for a different camera calibration (e.g., if video_index_param was 1 for fog_video.mp4)
        mtx = np.array([[1.15396467e+03, 0.00000000e+00, 6.69708251e+02],[0.00000000e+00, 1.14802823e+03, 3.85661017e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist = np.array([[-2.41026561e-01, -5.30262184e-02, -1.15775369e-03, -1.27924043e-04, 2.66417032e-02]])
        image = cv2.undistort(input_frame, mtx, dist, None, mtx) 
        top_left = [240, 270]
        top_right = [385, 270]
        bottom_right = [685, 402]
        bottom_left = [0, 402]
    
    src_mask = np.array([[(top_left[0], top_left[1]), (top_right[0], top_right[1]),
                           (bottom_right[0], bottom_right[1]), (bottom_left[0], bottom_left[1])]], np.int32)
    dst_mask = np.array([[(bottom_left[0], 0), (bottom_right[0], 0),
                           (bottom_right[0], bottom_right[1]), (bottom_left[0], bottom_left[1])]], np.int32)

    binary_image_raw = thresholding_pipeline(image, s_thresh=(90, 255))
    
    binary_warped_for_lines, M_transform = perspective_transform(binary_image_raw, src_mask, dst_mask)
    Minv = cv2.getPerspectiveTransform(dst_mask.astype(np.float32), src_mask.astype(np.float32))

    left_fit, right_fit, _, _, _, _ = sliding_windows(binary_warped_for_lines, nwindows=9)

    if left_fit is None or right_fit is None:
        # No usable lane lines: show the raw frame, clear the overlay and flag lane detection as unavailable
        # (the steering logic refuses to steer blind in that case).
        road = np.zeros_like(image)
        lane_detection_ok = False
        started = 1
        cv2.putText(image, "Lane lines not found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return image
    lane_detection_ok = True

    lane_overlay = project_lanelines(binary_warped_for_lines, image.shape, left_fit, right_fit, Minv)
    road = lane_overlay # Update global road variable

    # Radius of curvature in metres (average of the two lane lines; infinite = straight)
    radius_of_curvature = float('inf')
    if len(left_fit) == 3 and len(right_fit) == 3:
        height_px = binary_warped_for_lines.shape[0]
        xm_per_pix, ym_per_pix = lane_scale(left_fit, right_fit, height_px)
        radii = [curvature_radius_m(f, height_px - 1, xm_per_pix, ym_per_pix) for f in (left_fit, right_fit)]
        finite = [r for r in radii if np.isfinite(r)]
        if finite:
            radius_of_curvature = float(np.mean(finite))

    final_output_frame = cv2.addWeighted(image, 1, road, 0.5, 0) 
    curvature_text = ("Curvature: ~straight" if radius_of_curvature > 5000
                      else f"Curvature radius: {radius_of_curvature:.0f} m (approx.)")
    cv2.putText(final_output_frame, curvature_text, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    started = 1 
    return final_output_frame

def estimate_distance(bbox_width, bbox_height):
    """
    Estimates the distance of an object based on its bounding box width.
    """
    if bbox_width > 0:
        distance = (AVERAGE_CAR_WIDTH_METERS * FOCAL_LENGTH_PIXELS) / bbox_width
        return distance
    else:
        return 0.0

# --- Main execution block for combined detection ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # Not strictly necessary for Linux, but harmless

    # Whatever happens next (crash, Ctrl-C, SIGTERM, normal exit) the outputs end up in a safe state.
    atexit.register(safe_shutdown)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))   # sys.exit -> runs atexit handlers

    parser = argparse.ArgumentParser(description="Driver drowsiness detection with assisted pull-over.")
    parser.add_argument("--driver-cam", type=int, default=0, help="index of the driver-facing camera (default 0)")
    parser.add_argument("--road-cam", type=int, default=1, help="index of the road-facing camera (default 1)")
    parser.add_argument("--model", default=str(MODELS_DIR / "best_2.pt"), help="drowsiness model weights (default models/best_2.pt)")
    cli = parser.parse_args()

    # --- Camera Initialization ---
    # Camera for drowsiness detection (e.g., driver-facing)
    # On Raspberry Pi, camera IDs might vary. 0, 1 are common for USB webcams.
    # For Raspberry Pi Camera Module, you might need to use specific backends or picamera2.
    cap_drowsiness = cv2.VideoCapture(cli.driver_cam) 
    if not cap_drowsiness.isOpened():
        print(f"Error: Could not open drowsiness detection camera (ID {cli.driver_cam}). Please check camera connection.")
        sys.exit(1)

    # Camera for road/lane/car detection (e.g., front-facing)
    cap_road = cv2.VideoCapture(cli.road_cam) 
    if not cap_road.isOpened():
        print(f"Error: Could not open road detection camera (ID {cli.road_cam}). Please check camera connection or ID.")
        cap_drowsiness.release()
        sys.exit(1)

    print("Cameras initialized successfully.")

    try:
        # Load the drowsiness detection YOLO model
        # Ensure 'best_2.pt' is in the same directory as the script or provide full path
        model_drowsiness = YOLO(cli.model)
        # Never assume the class order: read it from the model itself
        AWAKE_CLASS_ID, DROWSY_CLASS_ID = resolve_class_ids(model_drowsiness.names)
        print(f"Drowsiness model '{cli.model}' loaded (awake={AWAKE_CLASS_ID}, drowsy={DROWSY_CLASS_ID}).")
    except Exception as e:
        print(f"Error loading drowsiness model: {e}")
        print("Please ensure 'ultralytics' is installed and the model file exists, is valid, and has 'awake' and 'drowsy' classes.")
        cap_drowsiness.release()
        cap_road.release()
        sys.exit(1)

    try:
        # Load the car detection YOLO model (yolov8n.pt for general object detection)
        # Ensure 'yolov8n.pt' is in the same directory as the script or provide full path
        model_lane = YOLO('yolov8n.pt') 
        print("Car detection model 'yolov8n.pt' loaded successfully.")
    except Exception as e:
        print(f"Error loading car detection model: {e}")
        print("Please ensure 'ultralytics' is installed and 'yolov8n.pt' exists and is a valid model file.")
        cap_drowsiness.release()
        cap_road.release()
        sys.exit(1)

    print("Starting combined detection. Press 'q' to quit.")

    # Define color ranges for the lane polygons (BGR format) for checking car position
    # These are adjusted for potential blending with the original video frame
    # These are BGR ranges that will check the `road` (lane overlay) image.
    purple_lower = np.array([80, 0, 80])   # Adjusted for blended purple
    purple_upper = np.array([200, 100, 200]) 
    
    yellow_lower = np.array([0, 100, 100]) # Adjusted for blended yellow
    yellow_upper = np.array([100, 255, 255])
    
    green_lower = np.array([0, 80, 0])     # Adjusted for blended green
    green_upper = np.array([100, 255, 100])

    # Target display resolution for individual camera feeds (before stacking)
    # Using 640x360 for each to keep the total width manageable on most screens (1280x360)
    # Adjust as needed based on your monitor's resolution.
    DISPLAY_WIDTH_PER_CAM = 640
    DISPLAY_HEIGHT_PER_CAM = 360

    while True:
        current_time = time.time() # Get current time for all timer logic

        # Read frames from cameras
        ret_drowsy, frame_drowsiness = cap_drowsiness.read()
        ret_road, frame_road = cap_road.read()

        # Check if all frames were read successfully
        if not (ret_drowsy and ret_road):
            print("Failed to read one or more frames. Exiting.")
            break

        # Resize frames for consistent display and model input
        # Drowsiness frame for model input (YOLO expects certain sizes, typically 640 for detection)
        resized_frame_drowsiness_for_model = cv2.resize(frame_drowsiness, (640, 384))
        
        # Road frame for lane detection processing (can be higher resolution for better lane accuracy)
        resized_for_lane_processing = cv2.resize(frame_road, (1280, 720))

        # Run drowsiness detection on its dedicated frame (resized for model)
        drowsiness_results = model_drowsiness(resized_frame_drowsiness_for_model, verbose=False)[0] 
        
        # NOTE: the car_in_*_lane flags are deliberately NOT reset here. The steering logic below reads the
        # values computed on the PREVIOUS frame; they are reset further down, just before they are recomputed.
        # (Resetting them at the top of the loop made the "car in right lane" safeguard always read False.)

        # Assume awake unless drowsiness is detected
        drowsiness_flag = 1  
        if drowsiness_results.boxes:
            for box in drowsiness_results.boxes:
                class_id = int(box.cls[0]) 
                if class_id == DROWSY_CLASS_ID:
                    drowsiness_flag = 0  # Drowsiness detected
                    break

        # --- Drowsiness Timer Logic, Buzzer Flag and LED Flag ---
        if drowsiness_flag == 0: # If currently drowsy
            if drowsy_start_time is None: # If drowsiness just started
                drowsy_start_time = current_time
            
            # Buzzer flag is set immediately when drowsiness is detected
            buzzer_flag = True 
            if GPIO_AVAILABLE: GPIO.output(BUZZER_PIN, GPIO.HIGH) # Activate physical buzzer for continuous alert

            # Start timer for LED only when buzzer_flag is first activated
            if buzzer_on_time is None:
                buzzer_on_time = current_time

            # LED flag is set if drowsy for more than 3 seconds after buzzer activation
            if (current_time - buzzer_on_time) > 3.0: 
                led_flag = True
                if GPIO_AVAILABLE: GPIO.output(LED_PIN, GPIO.HIGH) # Activate physical LED
            else:
                led_flag = False # Keep LED off if not yet 3 seconds since buzzer activated
                if GPIO_AVAILABLE: GPIO.output(LED_PIN, GPIO.LOW) # Deactivate physical LED
        else: # If currently awake
            drowsy_start_time = None # Reset overall drowsiness timer
            buzzer_on_time = None    # Reset buzzer activation timer
            buzzer_flag = False      # Turn off buzzer flag
            led_flag = False         # Turn off LED flag
            # Only turn off buzzer if it's not being used by the RPM momentary beep
            if GPIO_AVAILABLE and not rpm_buzz_active:
                GPIO.output(BUZZER_PIN, GPIO.LOW) # Deactivate physical buzzer
            if GPIO_AVAILABLE: # Ensure LED is off when awake
                GPIO.output(LED_PIN, GPIO.LOW)    # Deactivate physical LED

        # --- Motor Flag Logic ---
        # Motor is 1 by default, turns to 0 after 3 seconds of LED being on, and reverts to 1 when LED is off.
        # This logic assumes the motor is active (on) when motor_flag is 1, and inactive (off) when 0.
        # Relay polarity is handled by set_motor() via MOTOR_ACTIVE_HIGH (top of file).
        if led_flag:
            if led_on_time is None: # If LED just turned on
                led_on_time = current_time
            # If LED has been on for more than 3 seconds, turn motor off
            if (current_time - led_on_time) > 3.0:
                motor_flag = 0 
                set_motor(False)  # Deactivate motor
            else:
                motor_flag = 1 # Keep motor on while LED is on but waiting for 3 seconds
                set_motor(True)   # Activate motor
        else: # If LED is off
            led_on_time = None # Reset LED on timer
            motor_flag = 1     # Turn motor back on
            set_motor(True)   # Activate motor

        # --- Steering Angle Logic ---
        # Check if we are currently in a 180-degree cooldown period
        in_cooldown = False
        if steering_angle_180_cooldown_start_time is not None:
            if (current_time - steering_angle_180_cooldown_start_time) < 3.0:
                in_cooldown = True
            else:
                # Cooldown period has expired
                steering_angle_180_cooldown_start_time = None # Reset cooldown timer

        if in_cooldown:
            steering_angle = 180 # Force 180 during cooldown
            led_active_for_steering_time = None # Reset other timers
            steering_angle_210_start_time = None
        elif drowsiness_flag == 0 and lane_detection_ok and not car_in_right_outer_lane:
            # Drowsy, lane lines found and no car in the right outer lane (and not in cooldown)
            if led_flag:
                if led_active_for_steering_time is None:
                    # LED just became True, start the timer for initial 2-sec delay
                    led_active_for_steering_time = current_time
                    steering_angle = 180 # Keep at 180 during this initial delay
                elif (current_time - led_active_for_steering_time) >= 2.0:
                    # Initial 2-second LED active delay met, now manage the 210-degree phase
                    if steering_angle_210_start_time is None:
                        # Steering angle should now go to 210, start its own 2-second timer
                        steering_angle = 210
                        steering_angle_210_start_time = current_time
                    elif (current_time - steering_angle_210_start_time) >= 2.0:
                        # Steering angle has been at 210 for 2+ seconds, revert to 180
                        steering_angle = 180
                        steering_angle_210_start_time = None # Reset for next 210 cycle
                        steering_angle_180_cooldown_start_time = current_time # Start 180-degree cooldown
                    # Else: steering_angle is 210 and hasn't been 210 for 2 seconds yet, keep it at 210
            else:
                # Drowsy and no car, but LED is NOT on (yet, or no longer)
                steering_angle = 180
                led_active_for_steering_time = None # Reset timer if LED is off
                steering_angle_210_start_time = None # Reset the 210 duration timer
                steering_angle_180_cooldown_start_time = None # No cooldown if conditions not met
        else:
            # Not drowsy, OR lane lines not found, OR car in right outer lane: do not steer
            steering_angle = 180
            led_active_for_steering_time = None # Reset timer
            steering_angle_210_start_time = None # Reset the 210 duration timer
            steering_angle_180_cooldown_start_time = None # No cooldown if conditions not met
        
        # --- Apply steering angle to physical servo if GPIO is available ---
        if GPIO_AVAILABLE:
            # Convert angle to servo duty cycle.
            # Typical hobby servo expects a PWM signal. Angle mapping:
            # 0 degrees physical servo = ~2.5% duty cycle
            # 90 degrees physical servo = ~7.5% duty cycle (center)
            # 180 degrees physical servo = ~12.5% duty cycle
            # These values need to be CALIBRATED for your specific servo and mechanical setup.
            
            servo_angle_for_physical_servo = 90 # Default physical servo angle for 180 steering_angle
            if steering_angle == 180:
                servo_angle_for_physical_servo = 90  # Center position for a hobby servo
            elif steering_angle == 210:
                servo_angle_for_physical_servo = 100 # A small turn to the right
            # Add more `elif` conditions if you want more specific steering_angle values mapped to servo angles.

            # Convert physical servo angle (0-180) to PWM duty cycle (2.5-12.5)
            duty_cycle = (servo_angle_for_physical_servo / 18.0) + 2.5
            servo.ChangeDutyCycle(duty_cycle)

        # --- RPM Calculation ---
        # Calculate RPM every `RPM_UPDATE_INTERVAL_MS`
        if (current_time - rpm_last_update_time) * 1000 >= RPM_UPDATE_INTERVAL_MS:
            # Calculate RPM: (rotations / seconds_in_interval) * 60 seconds/minute
            seconds_in_interval = RPM_UPDATE_INTERVAL_MS / 1000.0
            rpm = (rpm_rotations / seconds_in_interval) * 60.0
            current_rpm = rpm
            rpm_rotations = 0 # Reset rotations for the next interval
            rpm_last_update_time = current_time # Update last update time

            # Calculate linear speed of the wheel
            # Circumference = 2 * pi * R = pi * D
            # Speed (m/s) = (RPM * Circumference in meters) / 60 seconds/minute
            # Speed (m/s) = (RPM * pi * D_meters) / 60
            wheel_speed_mps = (current_rpm * math.pi * WHEEL_DIAMETER_MM / 1000.0) / 60.0


        # --- Manage RPM Momentary Buzzer ---
        # This makes the buzzer beep for a short duration when RPM sensor triggers,
        # but only if the main drowsiness buzzer is NOT active.
        if rpm_buzz_active and GPIO_AVAILABLE:
            if (current_time - rpm_buzz_start_time) * 1000 > RPM_BUZZ_DURATION_MS:
                # Only turn off the buzzer if the drowsiness flag is NOT actively keeping it on
                if not buzzer_flag:
                    GPIO.output(BUZZER_PIN, GPIO.LOW) # Turn off buzzer
                rpm_buzz_active = False # Reset flag for next beep
            # If main buzzer_flag (drowsiness) is active, it will override this turning off
            # by immediately setting the pin HIGH again in the drowsiness logic above.


        # Get the annotated frame from drowsiness detection and resize for display
        annotated_frame_drowsiness = drowsiness_results.plot() 
        annotated_frame_drowsiness_display = cv2.resize(annotated_frame_drowsiness, (DISPLAY_WIDTH_PER_CAM, DISPLAY_HEIGHT_PER_CAM))


        # Determine drowsiness status text and color
        if drowsiness_flag == 1:
            display_text = "Driver is AWAKE!"
            car_in_left_outer_lane = car_in_ego_lane = car_in_right_outer_lane = False   # no road analysis while awake
            text_color = (0, 255, 0) # Green
            # When awake, stack drowsiness frame and a black placeholder for road
            road_display_placeholder = np.zeros((DISPLAY_HEIGHT_PER_CAM, DISPLAY_WIDTH_PER_CAM, 3), dtype=np.uint8)
            cv2.putText(road_display_placeholder, "Road Detection Inactive (Driver Awake)", (50, DISPLAY_HEIGHT_PER_CAM // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            combined_main_views = np.hstack((annotated_frame_drowsiness_display, road_display_placeholder))

        else: # drowsiness_flag is 0 (drowsy)
            display_text = "Driver is DROWSY! ALERT!"
            text_color = (0, 0, 255) # Red

            # Reset the per-frame lane flags now (the steering logic above has already used the previous frame's values)
            car_in_left_outer_lane = False
            car_in_ego_lane = False
            car_in_right_outer_lane = False

            # --- Execute Lane and Car Detection if Drowsy ---
            # Process the frame for lane detection
            lane_detection_output_frame = process_single_frame_for_lanes_and_cars(resized_for_lane_processing, video_index)
            # Resize it for display
            lane_detection_output_frame_display = cv2.resize(lane_detection_output_frame, (DISPLAY_WIDTH_PER_CAM, DISPLAY_HEIGHT_PER_CAM))


            # Now, run car detection using the car detection model on the same resized frame
            # The result object contains detected boxes, masks, etc.
            results_cars = model_lane(resized_for_lane_processing, verbose=False)[0]

            # Define color ranges for the lane polygons (BGR format) for checking car position
            # These are adjusted for potential blending with the original video frame
            # (already defined above in the global scope for clarity and reuse)

            # Process YOLOv8 car detection results to update flags and draw boxes
            if results_cars.boxes:
                for box in results_cars.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
                    conf = float(box.conf[0]) # Confidence score
                    cls = int(box.cls[0]) # Class ID

                    # Check if the detected object is a 'car' and confidence is above threshold
                    if model_lane.names[cls] == 'car' and conf >= 0.5:
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1

                        # Shrink the bounding box slightly for cleaner lane color sampling
                        box_shrink_factor = 0.15 
                        shrink_w_amount = int(bbox_width * box_shrink_factor / 2)
                        shrink_h_amount = int(bbox_height * box_shrink_factor / 2)

                        new_x1 = x1 + shrink_w_amount
                        new_y1 = y1 + shrink_h_amount
                        new_x2 = x2 - shrink_w_amount
                        new_y2 = y2 - shrink_h_amount

                        # Ensure new coordinates are within valid image bounds
                        new_x1 = max(0, new_x1)
                        new_y1 = max(0, new_y1)
                        new_x2 = min(resized_for_lane_processing.shape[1], new_x2)
                        new_y2 = min(resized_for_lane_processing.shape[0], new_y2)

                        # Define a small region at the bottom-center of the bounding box to check lane color
                        check_region_height = int(bbox_height * 0.1) # Check bottom 10% of box height
                        check_region_width = int(bbox_width * 0.5) # Check middle 50% of box width

                        center_x_bb = (new_x1 + new_x2) // 2
                        bottom_y_bb = new_y2 # Use bottom of the box for lane check

                        # Define the checking window coordinates in the original image space
                        check_x_start = max(0, center_x_bb - check_region_width // 2)
                        check_x_end = min(road.shape[1], center_x_bb + check_region_width // 2)
                        check_y_start = max(0, bottom_y_bb - check_region_height) # Start from slightly above the bottom
                        check_y_end = min(road.shape[0], bottom_y_bb + 1) # Go to the very bottom, inclusive

                        current_car_lane_color = None 

                        # Only proceed if the 'road' overlay has been initialized and the check region is valid
                        if started and check_y_end > check_y_start and check_x_end > check_x_start:
                            sub_road_img = road[check_y_start:check_y_end, check_x_start:check_x_end]

                            if sub_road_img.size > 0: # Ensure the sub-image is not empty
                                distance = estimate_distance(bbox_width, y2 - y1) # Estimate distance for this car

                                # Check for yellow (Ego Lane) first, as it's the primary lane
                                mask_yellow = cv2.inRange(sub_road_img, yellow_lower, yellow_upper)
                                if np.any(mask_yellow > 0) and distance < CAR_MAX_DISTANCE_M: # If yellow pixels detected and car is close
                                    current_car_lane_color = (0, 255, 255) # Yellow color for box
                                    car_in_ego_lane = True # Set global flag

                                # Check for purple (Left Outer Lane) if not in ego lane already
                                if current_car_lane_color is None: # Only check if not already assigned to ego lane
                                    mask_purple = cv2.inRange(sub_road_img, purple_lower, purple_upper)
                                    if np.any(mask_purple > 0) and distance < CAR_MAX_DISTANCE_M: # If purple pixels detected and car is close
                                        current_car_lane_color = (255, 0, 255) # Magenta color for box
                                        car_in_left_outer_lane = True # Set global flag

                                # Check for green (Right Outer Lane) if not in ego or left outer lane already
                                if current_car_lane_color is None: # Only check if not already assigned to ego or left lane
                                    mask_green = cv2.inRange(sub_road_img, green_lower, green_upper)
                                    if np.any(mask_green > 0) and distance < CAR_MAX_DISTANCE_M: # If green pixels detected and car is close
                                        current_car_lane_color = (0, 255, 0) # Green color for box
                                        car_in_right_outer_lane = True # Set global flag
                                    
                        # Only draw bounding box and labels if the car is detected within a lane AND distance is within range
                        if current_car_lane_color is not None:
                            label = f'{model_lane.names[cls]} {conf:.2f}' # Label with class name and confidence
                            # Draw on the `lane_detection_output_frame` (which is the high-res frame)
                            cv2.rectangle(lane_detection_output_frame, (new_x1, new_y1), (new_x2, new_y2), current_car_lane_color, 2) 
                            cv2.putText(lane_detection_output_frame, label, (new_x1, new_y1 - 10), # Put label above box
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_car_lane_color, 2)
                            
                            distance = estimate_distance(bbox_width, y2 - y1) # Re-calculate distance for display
                            distance_label = f'Distance: {distance:.2f}m' # Distance label
                            cv2.putText(lane_detection_output_frame, distance_label, (new_x1, new_y2 + 20), # Put distance below box
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Blue text for distance
            
            # Recalculate lane_detection_output_frame_display after drawing car boxes
            lane_detection_output_frame_display = cv2.resize(lane_detection_output_frame, (DISPLAY_WIDTH_PER_CAM, DISPLAY_HEIGHT_PER_CAM))
            combined_main_views = np.hstack((annotated_frame_drowsiness_display, lane_detection_output_frame_display))

            # Display the status of the lane flags on the right side of the combined display (relative to the stacked views)
            # Adjust these coordinates based on the combined image width (now 2 * DISPLAY_WIDTH_PER_CAM)
            cv2.putText(combined_main_views, f'Car in Left Outer: {car_in_left_outer_lane}', (DISPLAY_WIDTH_PER_CAM + 20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_main_views, f'Car in Ego Lane: {car_in_ego_lane}', (DISPLAY_WIDTH_PER_CAM + 20, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_main_views, f'Car in Right Outer: {car_in_right_outer_lane}', (DISPLAY_WIDTH_PER_CAM + 20, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add drowsiness status text to the appropriate side of the combined display
        cv2.putText(combined_main_views, display_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
        
        # Display the Buzzer, LED, Steering Angle, and Motor flag status
        cv2.putText(combined_main_views, f'Buzzer Flag: {buzzer_flag}', (50, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA) 
        cv2.putText(combined_main_views, f'LED Flag: {led_flag}', (50, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA) 
        cv2.putText(combined_main_views, f'Steering Angle: {steering_angle:.2f}', (50, 240), # Display steering angle
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 0, 255), 2, cv2.LINE_AA) # Purple color for Steering Angle
        cv2.putText(combined_main_views, f'Motor Flag: {motor_flag}', (50, 270), # Display motor flag
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA) # Greenish color for Motor Flag
        # Display RPM
        cv2.putText(combined_main_views, f'RPM: {current_rpm:.1f}', (50, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA) # Yellow color for RPM
        # Display Wheel Speed
        cv2.putText(combined_main_views, f'Speed: {wheel_speed_mps:.2f} m/s', (50, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA) # Cyan color for Speed


        # Show the final combined output
        cv2.imshow('Combined Detection (Drowsiness & Lane/Car)', combined_main_views)

        # Handle 'q' key press to exit
        key = cv2.waitKey(1) & 0xFF 
        if key == ord('q'):
            print(" 'q' key pressed. Exiting application.")
            break

    # Release all camera capture objects and close all OpenCV windows
    cap_drowsiness.release()
    cap_road.release()
    cv2.destroyAllWindows() 
    
    # --- Clean up GPIO if it was initialized ---
    safe_shutdown()   # buzzer/LED off, steering centred, motor relay OFF, GPIO released
    
    print("Combined detection application finished.")
