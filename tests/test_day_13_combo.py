#!/usr/bin/env python3
"""Unit tests for day_13_combo.py — document correction pipeline.

Run with:
    python -m unittest tests/test_day_13_combo.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Force source-level import (bypass stale .pyc cache)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "experiments" / "day_13_combo.py"
spec = importlib.util.spec_from_file_location("day_13_combo", SRC_PATH)
D13 = importlib.util.module_from_spec(spec)

# Headless matplotlib (must be set *before* the module code runs)
import matplotlib
matplotlib.use("Agg")

spec.loader.exec_module(D13)


# ================================================================
# Helpers
# ================================================================
def _blank_bgr(h=100, w=100):
    return np.full((h, w, 3), 128, dtype=np.uint8)


# ================================================================
# is_supported_image
# ================================================================
class TestIsSupportedImage(unittest.TestCase):
    def test_nonexistent(self):
        self.assertFalse(D13.is_supported_image(Path("/nonexistent/x.jpg")))

    def test_invalid_ext(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            p = Path(f.name)
        try:
            p.write_text("x")
            self.assertFalse(D13.is_supported_image(p))
        finally:
            p.unlink(missing_ok=True)

    def test_heic_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".heic", delete=False) as f:
            p = Path(f.name)
        try:
            p.write_text("x")
            self.assertFalse(D13.is_supported_image(p))
        finally:
            p.unlink(missing_ok=True)

    def test_valid_content(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            p = Path(f.name)
        try:
            cv2.imwrite(str(p), _blank_bgr(10, 10))
            self.assertTrue(D13.is_supported_image(p, verify_content=True))
        finally:
            p.unlink(missing_ok=True)

    def test_corrupt_content(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            p = Path(f.name)
        try:
            p.write_bytes(b"not a valid jpeg")
            self.assertFalse(D13.is_supported_image(p, verify_content=True))
        finally:
            p.unlink(missing_ok=True)

    def test_directory_not_file(self):
        """A directory path should return False (not a regular file)."""
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(D13.is_supported_image(Path(d)))

    def test_empty_file(self):
        """A zero-byte file with .jpg extension should fail content check."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            p = Path(f.name)
        try:
            # Extension check passes, content check should fail
            self.assertTrue(D13.is_supported_image(p, verify_content=False))
            self.assertFalse(D13.is_supported_image(p, verify_content=True))
        finally:
            p.unlink(missing_ok=True)


# ================================================================
# add_salt_pepper_noise
# ================================================================
class TestSaltPepper(unittest.TestCase):
    def test_shape_preserved(self):
        img = _blank_bgr()
        out = D13.add_salt_pepper_noise(img, 0.02)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, img.dtype)

    def test_noise_fraction(self):
        img = np.full((200, 200, 3), 128, np.uint8)
        np.random.seed(42)
        out = D13.add_salt_pepper_noise(img, 0.02)
        self.assertAlmostEqual(np.sum(out == 255) / img.size, 0.01, delta=0.005)
        self.assertAlmostEqual(np.sum(out == 0) / img.size, 0.01, delta=0.005)

    def test_zero_prob(self):
        img = _blank_bgr()
        self.assertTrue(np.array_equal(D13.add_salt_pepper_noise(img, 0.0), img))

    def test_prob_one_all_noisy(self):
        """prob=1.0 means every pixel becomes either 0 or 255."""
        img = np.full((50, 50, 3), 128, np.uint8)
        out = D13.add_salt_pepper_noise(img, 1.0)
        self.assertEqual(np.sum(out == 128), 0)
        self.assertEqual(np.sum(out == 0) + np.sum(out == 255), out.size)


# ================================================================
# denoise_median
# ================================================================
class TestDenoiseMedian(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(D13.denoise_median(_blank_bgr(), 3).shape, (100, 100, 3))

    def test_removes_noise(self):
        img = np.full((100, 100, 3), 128, np.uint8)
        np.random.seed(0)
        noisy = D13.add_salt_pepper_noise(img, 0.05)
        clean = D13.denoise_median(noisy, 5)
        n_before = np.sum(noisy == 0) + np.sum(noisy == 255)
        n_after = np.sum(clean == 0) + np.sum(clean == 255)
        self.assertLess(n_after, n_before * 0.3)

    def test_ksize_one_identity(self):
        """ksize=1 median is near-identity (every pixel is its own median)."""
        img = _blank_bgr()
        out = D13.denoise_median(img, 1)
        self.assertTrue(np.array_equal(img, out))


# ================================================================
# smooth_gaussian
# ================================================================
class TestSmoothGaussian(unittest.TestCase):
    def test_shape(self):
        gray = np.full((50, 50), 128, np.uint8)
        self.assertEqual(D13.smooth_gaussian(gray, (3, 3)).shape, gray.shape)

    def test_reduces_variance(self):
        i, j = np.indices((200, 200))
        bw = (((i // 25) % 2) ^ ((j // 25) % 2)).astype(np.uint8) * 255
        v0 = cv2.Laplacian(bw, cv2.CV_64F).var()
        v1 = cv2.Laplacian(D13.smooth_gaussian(bw, (15, 15)), cv2.CV_64F).var()
        self.assertLess(v1, v0 / 2)


# ================================================================
# extract_edges
# ================================================================
class TestExtractEdges(unittest.TestCase):
    def setUp(self):
        doc = D13._make_test_document()
        gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)
        self.blurred = D13.smooth_gaussian(gray, D13.GAUSSIAN_KSIZE)

    def test_binary(self):
        edges = D13.extract_edges(self.blurred, 50, 150)
        self.assertTrue(set(np.unique(edges)).issubset({0, 255}))

    def test_threshold_density(self):
        lo = cv2.countNonZero(D13.extract_edges(self.blurred, 30, 90))
        hi = cv2.countNonZero(D13.extract_edges(self.blurred, 150, 250))
        self.assertLessEqual(hi, lo)

    def test_reasonable_count(self):
        n = cv2.countNonZero(D13.extract_edges(self.blurred, 50, 150))
        self.assertGreater(n, 5000)
        self.assertLess(n, 300000)

    def test_uniform_image_no_edges(self):
        """A perfectly uniform image should produce (near) zero edges."""
        flat = np.full((100, 100), 128, dtype=np.uint8)
        edges = D13.extract_edges(flat, 50, 150)
        self.assertEqual(cv2.countNonZero(edges), 0)

    def test_low_geq_high(self):
        """low >= high is technically valid (Canny clamps internally)."""
        edges = D13.extract_edges(self.blurred, 200, 50)
        self.assertTrue(set(np.unique(edges)).issubset({0, 255}))
        self.assertEqual(edges.shape, self.blurred.shape)


# ================================================================
# find_largest_quadrilateral
# ================================================================
class TestFindQuad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        doc = D13._make_test_document()
        denoised = D13.denoise_median(doc, D13.MEDIAN_KSIZE)
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        blurred = D13.smooth_gaussian(gray, D13.GAUSSIAN_KSIZE)
        cls.edges = D13.extract_edges(blurred, D13.CANNY_LOW, D13.CANNY_HIGH)

    def test_finds_quad(self):
        quad = D13.find_largest_quadrilateral(self.edges)
        self.assertIsNotNone(quad)
        self.assertEqual(quad.shape, (4, 2))

    def test_large_area(self):
        quad = D13.find_largest_quadrilateral(self.edges)
        area = cv2.contourArea(quad)
        img_area = self.edges.shape[0] * self.edges.shape[1]
        self.assertGreater(area, img_area * 0.3)

    def test_none_on_blank(self):
        self.assertIsNone(D13.find_largest_quadrilateral(np.zeros((100, 100), np.uint8)))

    def test_none_when_no_quad_contours(self):
        """Edge map with only non-quadrilateral contours → return None."""
        canvas = np.zeros((200, 200), dtype=np.uint8)
        # Draw three filled circles — no 4-vertex contour possible
        cv2.circle(canvas, (60, 100), 30, 255, 2)
        cv2.circle(canvas, (140, 60), 25, 255, 2)
        cv2.circle(canvas, (100, 150), 20, 255, 2)
        self.assertIsNone(D13.find_largest_quadrilateral(canvas))

    def test_picks_largest_when_multiple_quads(self):
        """When several quadrilaterals exist, return the one with max area."""
        canvas = np.zeros((300, 300), dtype=np.uint8)
        # Small quad (area ≈ 2500)
        cv2.rectangle(canvas, (210, 210), (260, 260), 255, 2)
        # Large quad (area ≈ 40000) — dominant document boundary
        cv2.rectangle(canvas, (30, 30), (270, 270), 255, 2)
        # Medium quad (area ≈ 8100)
        cv2.rectangle(canvas, (180, 30), (270, 120), 255, 2)

        result = D13.find_largest_quadrilateral(canvas)
        self.assertIsNotNone(result)
        area = cv2.contourArea(result)
        self.assertGreater(area, 30000)  # picked the large one, not the small


# ================================================================
# _order_corners
# ================================================================
class TestOrderCorners(unittest.TestCase):
    def test_already_ordered(self):
        pts = np.float32([[0, 0], [100, 0], [100, 80], [0, 80]])
        np.testing.assert_allclose(D13._order_corners_tl_tr_br_bl(pts), pts)

    def test_shuffled(self):
        exp = np.float32([[0, 0], [100, 0], [100, 80], [0, 80]])
        shuf = np.float32([[100, 0], [0, 80], [100, 80], [0, 0]])
        np.testing.assert_allclose(D13._order_corners_tl_tr_br_bl(shuf), exp)

    def test_near_degenerate(self):
        """Almost-collinear points should still produce a (4,2) output."""
        pts = np.float32([[10, 100], [20, 100], [60, 99], [50, 101]])
        ordered = D13._order_corners_tl_tr_br_bl(pts)
        self.assertEqual(ordered.shape, (4, 2))


# ================================================================
# warp_document
# ================================================================
class TestWarpDocument(unittest.TestCase):
    def setUp(self):
        self.img = D13._make_test_document()
        self.src = np.float32([[100, 50], [1400, 80], [1300, 800], [200, 780]])

    def test_output_shape(self):
        w = D13.warp_document(self.img, self.src, (400, 500))
        self.assertEqual(w.shape, (500, 400, 3))

    def test_bgr_channels(self):
        w = D13.warp_document(self.img, self.src, (400, 500))
        self.assertEqual(w.ndim, 3)
        self.assertEqual(w.shape[2], 3)

    def test_warped_content_not_black(self):
        """Warped output must contain actual content, not just black pixels."""
        w = D13.warp_document(self.img, self.src, (400, 500))
        mean_val = np.mean(w)
        self.assertGreater(mean_val, 10, "Warped image is unexpectedly dark")


# ================================================================
# process_single_image
# ================================================================
class TestProcessSingleImage(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def _write(self, name, img):
        p = self.tmp / name
        cv2.imwrite(str(p), img)
        return p

    def test_synthetic_succeeds(self):
        p = self._write("doc.jpg", D13._make_test_document())
        r = D13.process_single_image(p)
        self.assertTrue(r["success"])
        self.assertGreater(r["area"], 0)
        self.assertIsNotNone(r["corners"])
        self.assertIsNotNone(r["warped"])
        self.assertGreaterEqual(len(r["steps_log"]), 5)

    def test_nonexistent(self):
        self.assertFalse(D13.process_single_image(Path("/nonexistent/x.jpg"))["success"])

    def test_unsupported_fmt(self):
        p = self.tmp / "x.txt"
        p.write_text("x")
        self.assertFalse(D13.process_single_image(p)["success"])

    def test_corrupt(self):
        p = self.tmp / "bad.jpg"
        p.write_bytes(b"not a jpeg")
        self.assertFalse(D13.process_single_image(p)["success"])

    def test_blank(self):
        p = self._write("blank.jpg", _blank_bgr(100, 100))
        self.assertFalse(D13.process_single_image(p)["success"])

    def test_dict_keys(self):
        p = self._write("doc.jpg", D13._make_test_document())
        r = D13.process_single_image(p)
        self.assertEqual(set(r.keys()), {"path", "success", "area", "corners", "warped", "steps_log"})

    def test_no_document_in_scene(self):
        """A natural photo without a document should fail gracefully."""
        # Generate a smooth gradient image — no sharp edges, no quad
        grad = np.linspace(0, 255, 200, dtype=np.uint8)
        img = np.tile(grad.reshape(200, 1), (1, 200, 1))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        p = self._write("landscape.jpg", img)
        r = D13.process_single_image(p)
        self.assertFalse(r["success"])
        self.assertIsNone(r["corners"])
        self.assertIsNone(r["warped"])


# ================================================================
# build_combo_report
# ================================================================
class TestBuildReport(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def _gray(self):
        doc = D13._make_test_document()
        return cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)

    def test_with_corners(self):
        g = self._gray()
        c = np.float32([[250, 170], [1350, 280], [1200, 1050], [380, 930]])
        self.assertIsNotNone(D13.build_combo_report(g, None, g, g, g, g, c))

    def test_without_corners(self):
        g = self._gray()
        self.assertIsNotNone(D13.build_combo_report(g, None, g, g,
            np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.uint8), None))


# ================================================================
# _make_test_document
# ================================================================
class TestMakeTestDocument(unittest.TestCase):
    def test_deterministic(self):
        """Same seed / same params → pixel-identical output."""
        a = D13._make_test_document()
        b = D13._make_test_document()
        self.assertTrue(np.array_equal(a, b))

    def test_output_shape(self):
        doc = D13._make_test_document(480, 640)
        self.assertEqual(doc.shape, (480, 640, 3))

    def test_not_blank(self):
        """Synthetic document should have varied pixel values, not uniform."""
        doc = D13._make_test_document()
        self.assertGreater(np.std(doc), 20)  # non-trivial content


# ================================================================
# Config
# ================================================================
class TestConfig(unittest.TestCase):
    def test_no_heic(self):
        self.assertNotIn(".heic", D13.SUPPORTED_EXTENSIONS)

    def test_output_size(self):
        self.assertEqual(D13.OUTPUT_W, 2100)
        self.assertEqual(D13.OUTPUT_H, 2970)

    def test_canny_order(self):
        self.assertLess(D13.CANNY_LOW, D13.CANNY_HIGH)


# ================================================================
if __name__ == "__main__":
    unittest.main()
