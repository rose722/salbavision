# =============================================================================
# drowning.py  — Standalone drowning detection viewer
# =============================================================================
#
# Runs the Roboflow InferencePipeline and shows a live annotated window.
# No Flask, no Supabase — just detection + visual output.
#
# RUN:
#   py drowning.py
#   Press q in the window to stop.
#
# INSTALL (if not done yet):
#   pip install inference opencv-python pillow
#   (GPU) pip install inference-gpu
#
# =============================================================================

import os
import cv2
import numpy as np
from inference import InferencePipeline
from PIL import Image, ImageDraw, ImageFont

# =============================================================================
# CONFIG — change these to switch source or model
# =============================================================================

RF_API_KEY   = "yYf0oFRqVThzJtqnC6D4"
RF_MODEL_ID  = "aqw3rfaq3wcqrq2r/9"

# 0 = default webcam | path/to/video.mp4 | "rtsp://..."
# VIDEO_SOURCE = "rtsp://awts:888888@192.180.100.17:554/stream1"
VIDEO_SOURCE = r"C:\Github Projects\DROWNING_DETECTION_SYSTEM\salbavision-v2\videos\video_20260401_132916.mp4"

MAX_FPS = 60

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

CLASS_COLORS_BGR = {
    "drowning":     (0, 0, 255),
    "out of water": (0, 255, 0),
    "swimming":     (255, 0, 127),
}

# =============================================================================
# GPU SETUP
# =============================================================================

try:
    import torch
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = (
            "CUDAExecutionProvider,CPUExecutionProvider"
        )
        print(f"[GPU] {torch.cuda.get_device_name(0)} — using GPU")
    else:
        print("[GPU] No CUDA detected — running on CPU")
except ImportError:
    print("[GPU] PyTorch not installed")

# =============================================================================
# FONT
# =============================================================================

try:
    FONT = ImageFont.truetype(FONT_PATH, 20)
except OSError:
    FONT = ImageFont.load_default()

# =============================================================================
# PREDICTION CALLBACK
# =============================================================================

def handle_prediction(prediction_data, frame):
    if hasattr(frame, "image"):
        frame = frame.image
    if not isinstance(frame, np.ndarray):
        return

    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    for pred in prediction_data.get("predictions", []):
        if not isinstance(pred, dict):
            continue

        x, y = pred.get("x", 0), pred.get("y", 0)
        w, h = pred.get("width", 0), pred.get("height", 0)
        cls  = pred.get("class", "")
        conf = pred.get("confidence", 0.0)

        color_bgr = CLASS_COLORS_BGR.get(cls, (255, 255, 255))
        color_rgb = color_bgr[::-1]

        pt1 = (int(x - w / 2), int(y - h / 2))
        pt2 = (int(x + w / 2), int(y + h / 2))

        draw.rectangle([pt1, pt2], outline=color_rgb, width=6)

        label = f"{cls} ({conf:.2f})"
        tb = draw.textbbox((0, 0), label, font=FONT)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx = pt1[0]
        ly = pt1[1] - th - 4 if pt1[1] - th - 4 >= 0 else pt1[1] + 4
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=color_rgb)
        draw.text((lx + 3, ly + 2), label, font=FONT, fill="white")

    out = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cv2.imshow("Drowning Detection", out)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        os._exit(0)


# =============================================================================
# RUN
# =============================================================================

pipeline = InferencePipeline.init(
    api_key=RF_API_KEY,
    model_id=RF_MODEL_ID,
    video_reference=VIDEO_SOURCE,
    on_prediction=handle_prediction,
    max_fps=MAX_FPS,
)

pipeline.start()
pipeline.join()
