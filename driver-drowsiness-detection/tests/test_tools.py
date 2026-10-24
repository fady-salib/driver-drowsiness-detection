import importlib.util
import os
import tempfile
import unittest

import cv2
import numpy as np

import fakes

spec = importlib.util.spec_from_file_location("calibrate_road", os.path.join(fakes.REPO, "tools", "calibrate_road.py"))
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class CalibrateRoad(unittest.TestCase):
    def test_default_roi_matches_the_application(self):
        roi = fakes.load_module()["ROAD_ROI"]
        expected = [tuple(roi[k]) for k in ("top_left", "top_right", "bottom_right", "bottom_left")]
        self.assertEqual(tool.parse_roi(tool.DEFAULT_ROI), expected)

    def test_parse_roi_rejects_wrong_point_count(self):
        with self.assertRaises(ValueError):
            tool.parse_roi("1,2 3,4 5,6")

    def test_render_keeps_frame_size_and_marks_the_roi(self):
        frame = np.zeros((480, 640, 3), np.uint8)                      # any input size is resized to 1280x720
        annotated, warped = tool.render(frame, tool.parse_roi(tool.DEFAULT_ROI))
        self.assertEqual(annotated.shape, (720, 1280, 3))
        self.assertEqual(warped.shape, (720, 1280, 3))
        self.assertGreater(int(annotated.sum()), 0)                    # the polygon was drawn

    def test_straight_lines_become_vertical_in_the_birds_eye_view(self):
        roi = tool.parse_roi(tool.DEFAULT_ROI)
        # draw lane lines exactly along the left/right edges of the trapezoid
        img = np.zeros((720, 1280, 3), np.uint8)
        cv2.line(img, roi[0], roi[3], (255, 255, 255), 6)
        cv2.line(img, roi[1], roi[2], (255, 255, 255), 6)
        _, warped = tool.render(img, roi)
        gray = warped[:, :, 0]
        for row in (50, 300, 600):
            xs = np.nonzero(gray[row] > 128)[0]
            self.assertTrue(len(xs) > 0)
            self.assertLess(xs.min(), roi[3][0] + 12)                  # left line at the left edge of the ROI
            self.assertGreater(xs.max(), roi[2][0] - 12)               # right line at the right edge

    def test_cli_writes_an_output_file(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = os.path.join(d, "in.png"), os.path.join(d, "out.png")
            cv2.imwrite(src, fakes.yellow_road())
            tool.main(["--image", src, "--out", out])
            self.assertEqual(cv2.imread(out).shape, (360, 1280, 3))

    def test_missing_image_is_a_clean_error(self):
        with self.assertRaises(SystemExit):
            tool.main(["--image", "/nonexistent.png", "--out", "/tmp/x.png"])


if __name__ == "__main__":
    unittest.main()
