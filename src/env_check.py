"""
Environment Check
Verify that all dependencies for opencv-labs are working.

Usage:
    python src/env_check.py
    python src/env_check.py --video path/to/test.mp4

Checks:
    1. Python version, NumPy, OpenCV, Matplotlib, Pillow
    2. GUI: imshow / Trackbar / mouse callbacks
    3. Video I/O: VideoCapture (MP4), VideoWriter
    4. File system: project directory structure
    5. Performance: quick CPU benchmark
    6. GPU: OpenCV CUDA, PyTorch CUDA, OpenCL
    7. Summary
"""
import sys
import os
import time
import warnings
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SEP = "-" * 55


def header(title):
    print("\n" + SEP)
    print("  " + title)
    print(SEP)


def check(name, result, detail=""):
    icon = PASS if result else FAIL
    tail = "  -- " + detail if detail else ""
    print("  [%s] %s%s" % (icon, name, tail))
    return result


# ============================================================
# 1. Python & Core Libraries
# ============================================================
header("1. Python & Core Libraries")

py_ok = check("Python %s" % sys.version.split()[0],
              sys.version_info >= (3, 9),
              "path: %s" % sys.executable)

try:
    import numpy as np
    np_ok = check("NumPy %s" % np.__version__, True)
except ImportError:
    np_ok = check("NumPy", False, "pip install numpy")

try:
    import cv2
    cv_ver = cv2.__version__
    cv_ok = check("OpenCV %s" % cv_ver, True)
except ImportError:
    cv_ok = False
    check("OpenCV", False, "pip install opencv-python")

try:
    import matplotlib
    mpl_ok = check("Matplotlib %s" % matplotlib.__version__, True)
except ImportError:
    mpl_ok = check("Matplotlib", False, "pip install matplotlib")

try:
    from PIL import Image
    pil_ok = check("Pillow %s" % Image.__version__, True)
except ImportError:
    pil_ok = check("Pillow", False, "pip install Pillow")

core_ok = py_ok and np_ok and cv_ok and mpl_ok and pil_ok
if not core_ok:
    print("\n  Missing dependencies. Install with:")
    print("    pip install opencv-python numpy matplotlib Pillow")
    sys.exit(1)


# ============================================================
# 2. OpenCV Capabilities
# ============================================================
header("2. OpenCV Capabilities")

# --- GUI: imshow ---
gui_ok = False
try:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imshow("_test_", img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    gui_ok = True
    check("imshow / waitKey", True)
except cv2.error:
    check("imshow / waitKey", False,
          "GUI not available (headless build). Use matplotlib instead.")

# --- Trackbar ---
if gui_ok:
    try:
        cv2.namedWindow("_test_")
        cv2.createTrackbar("t", "_test_", 0, 100, lambda x: None)
        cv2.destroyWindow("_test_")
        check("Trackbar", True)
    except Exception:
        check("Trackbar", False, "Will use CLI args instead")
else:
    check("Trackbar", False, "Requires GUI")

# --- Mouse callback ---
if gui_ok:
    check("Mouse callback", True, "setMouseCallback available")
else:
    check("Mouse callback", False, "Will use hardcoded coordinates")

# --- Video: backend ---
has_videoio = hasattr(cv2, 'VideoCapture')
check("VideoCapture module", has_videoio,
      "videoio available" if has_videoio else "missing")

# --- Video: HEVC MP4 ---
hevc_ok = False
video_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
mp4_files = sorted(video_dir.glob("*.mp4"))
if mp4_files:
    f = str(mp4_files[0])
    cap = cv2.VideoCapture(f)
    ret, _ = cap.read()
    cap.release()
    hevc_ok = ret
    size_mb = os.path.getsize(f) / 1024 / 1024
    check("Video file read (%s)" % mp4_files[0].name, ret,
          "%.0f MB" % size_mb if ret else "decode failed")
else:
    check("Video file test", False,
          "No mp4 found in %s" % video_dir)

# --- VideoWriter ---
try:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = cv2.VideoWriter("/tmp/_env_test.mp4", fourcc, 1, (100, 100))
    out.write(img)
    out.release()
    check("VideoWriter (mp4v)", True)
except Exception:
    check("VideoWriter (mp4v)", False)

# --- putText / font ---
try:
    img = np.zeros((50, 200, 3), dtype=np.uint8)
    cv2.putText(img, "Test", (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    check("putText / font rendering", True)
except Exception:
    check("putText / font rendering", False)


# ============================================================
# 3. File System
# ============================================================
header("3. File System")

project_root = Path(__file__).resolve().parent.parent
check("project_root", project_root.exists(), str(project_root))

for sub in ["data/raw", "data/processed", "experiments",
            "notebooks", "docs", "src"]:
    p = project_root / sub
    check("  %s/" % sub, p.exists(),
          "found" if p.exists() else "create it with mkdir")


# ============================================================
# 4. Performance (CPU)
# ============================================================
header("4. Performance (CPU, 1920x1440)")

img = np.random.randint(0, 255, (1440, 1920, 3), dtype=np.uint8)

t0 = time.perf_counter()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t1 = time.perf_counter()

t2 = time.perf_counter()
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
t3 = time.perf_counter()

t4 = time.perf_counter()
edges = cv2.Canny(blurred, 50, 150)
t5 = time.perf_counter()

t6 = time.perf_counter()
_ = cv2.resize(img, (640, 480))
t7 = time.perf_counter()

print("  cvtColor:       %5.1f ms" % ((t1 - t0) * 1000))
print("  GaussianBlur:   %5.1f ms" % ((t3 - t2) * 1000))
print("  Canny:          %5.1f ms" % ((t5 - t4) * 1000))
print("  resize 640x480: %5.1f ms" % ((t7 - t6) * 1000))


# ============================================================
# 5. GPU / CUDA
# ============================================================
header("5. GPU / CUDA")

# --- OpenCV CUDA ---
try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_count > 0:
        check("OpenCV CUDA devices", True, "%d device(s)" % cuda_count)
        for i in range(cuda_count):
            cv2.cuda.setDevice(i)
            info = cv2.cuda.DeviceInfo()
            mem_mb = info.totalMemory() / 1024 / 1024
            cc = "%d.%d" % (info.majorVersion(), info.minorVersion())
            check("  GPU %d: %s" % (i, info.name()), True,
                  "CC %s, %.0f MB" % (cc, mem_mb))
    else:
        check("OpenCV CUDA", False, "No CUDA devices or OpenCV built without CUDA")
except cv2.error:
    check("OpenCV CUDA", False, "OpenCV built without CUDA support")
except Exception as e:
    check("OpenCV CUDA", False, str(e))

# --- PyTorch (optional) ---
try:
    import torch
    pt_ver = torch.__version__
    pt_cuda = torch.cuda.is_available()
    if pt_cuda:
        pt_dev = torch.cuda.device_count()
        pt_name = torch.cuda.get_device_name(0)
        pt_mem = torch.cuda.get_device_properties(0).total_mem
        detail = "CUDA available: %dx %s, %.1f GB" % (
            pt_dev, pt_name, pt_mem / 1024**3)
        check("PyTorch " + pt_ver, True, detail)
    else:
        check("PyTorch " + pt_ver, True, "CUDA NOT available (CPU only)")
except ImportError:
    print("  [ -- ] PyTorch: not installed (optional, not required)")
except Exception as e:
    check("PyTorch", False, str(e))

# --- OpenCL (Intel/AMD fallback) ---
try:
    ocl_available = cv2.ocl.haveOpenCL()
    if ocl_available:
        dev = cv2.ocl.Device.getDefault()
        if dev is not None and dev.name():
            check("OpenCL device", True, str(dev.name()))
        else:
            check("OpenCL device", False, "haveOpenCL()=True but no valid device")
    else:
        check("OpenCL device", False, "No OpenCL runtime found (haveOpenCL()=False)")
except Exception:
    check("OpenCL device", False, "OpenCL not available")


# ============================================================
# 6. Summary
# ============================================================
header("6. Summary")

all_ok = core_ok
print("  Core libraries:  %s" % ("ALL OK" if all_ok else "SOME MISSING"))
print("  GUI (imshow):    %s" % ("Available" if gui_ok else "HEADLESS - use matplotlib"))
print("  Video (MP4):     %s" % ("Decoding OK" if hevc_ok else "Not tested / failed"))
print("  Matplotlib:      %s" % ("Ready" if mpl_ok else "Not available"))

if not gui_ok:
    print()
    print("  NOTE: This environment is headless (no imshow).")
    print("    Use matplotlib.pyplot.imshow() for image display.")
    print("    Use for-loops + savefig() instead of Trackbar interaction.")

print()
print(SEP)
print("  Environment check complete.")
print(SEP)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     