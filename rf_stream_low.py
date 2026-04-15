# =============================================================================
# rf_stream_low.py
# Low-latency Roboflow stream bridge for Salbavision (CPU-optimized)
# =============================================================================
#
# Architecture (mirrors stream_bridge.py, but uses Roboflow get_model):
#
#   [capture thread]  reads VIDEO_SOURCE at native fps  → latest_frame
#   [detect  thread]  runs model.infer every FRAME_SKIP → latest_predictions
#   [display/encode]  draws cached boxes on every fresh frame, encodes JPEG once
#   [flask generate]  waits on threading.Event, yields cached JPEG bytes
#
# Why this is smoother than rf_stream_bridge.py on an i5 + Intel HD:
#   - Video is displayed at source fps regardless of inference speed
#   - Inference lags behind but boxes persist, so detection feels continuous
#   - JPEG encoded once per display frame, not re-encoded every MJPEG pull
#   - No busy-loop in generate(); blocked on an Event
#
# REQUIREMENTS
#   pip install inference opencv-python pillow flask supabase pygame
#
# RUN
#   python rf_stream_low.py
#   Stream → http://localhost:5001/video_feed
#   Alert  → http://localhost:5001/latest_alert
#   Status → http://localhost:5001/status
# =============================================================================

import os
import signal
import time
import threading
from collections import deque

import cv2
import numpy as np
import pygame
from flask import Flask, Response, jsonify, request
from inference import get_model
from PIL import Image, ImageDraw, ImageFont
from supabase import create_client, Client

# =============================================================================
# CONFIG
# =============================================================================

# --- Roboflow ---
RF_API_KEY  = "yYf0oFRqVThzJtqnC6D4"
RF_MODEL_ID = "aqw3rfaq3wcqrq2r/9"

# --- Video source ---
# VIDEO_SOURCE = "rtsp://awts11:12345678@192.168.1.16:554/stream1"
# VIDEO_SOURCE = r"C:\Users\eirmo\Documents\freelance_systems\salbavision\videos\IMG_1205.MOV"
VIDEO_SOURCE = r"C:\Users\SHEEN\OneDrive\Documents\salbavision\videos\IMG_1205.MOV"

# Loop video files (for dev / demo) so the stream never ends
LOOP_VIDEO_FILES = True

# --- Supabase ---
SUPABASE_URL = "https://yzohitznmgtzdkzyoztf.supabase.co"
SUPABASE_KEY = "sb_secret_Q8_z2vsv5-x-KxSk25AJjQ_ONbMzgKF"

# --- Audio ---
AUDIO_ENABLED = True
SIREN_FILE    = "siren.mp3"

# --- Detection thresholds (mirrors stream_bridge.py exactly) ---
BASE_CONF             = 0.25
DROWN_DRAW_THRESHOLD  = 0.40
OUT_DRAW_THRESHOLD    = 0.45
SWIM_DRAW_THRESHOLD   = 0.45
DROWN_ALERT_THRESHOLD = 0.50
DOMINANCE_MARGIN      = 0.21
HARD_SUPPRESS_MARGIN  = 0.05
MIN_AREA_RATIO        = 0.002

STATE_HISTORY_LEN     = 12
REQUIRED_DROWN_FRAMES = 2
ALARM_HOLD            = 5
ALERT_COOLDOWN        = 10

# --- Performance tuning (i5 + Intel HD defaults) ---
FRAME_SKIP        = 2       # run inference every Nth captured frame
DISPLAY_WIDTH     = 640     # resize before encode
JPEG_QUALITY      = 70
TARGET_DISPLAY_FPS = 20     # cap display/encode rate
RTSP_CHECK_TIMEOUT = 8
SOURCE_RETRY_DELAY = 5

# --- Debug ---
# Set True to bypass suppression and draw every raw detection (drowning.py style).
# Use this to confirm the model is returning "drowning" on IMG_1205 and to
# observe real confidence numbers before tuning the gates above.
DEBUG_PASSTHROUGH  = False
DEBUG_PRINT_EVERY  = 15     # print raw detections every N inferences (0 = off)

FONT_PATH = "C:/Windows/Fonts/arial.ttf"

CLASS_COLORS_BGR = {
    "drowning":     (0, 0, 255),
    "out of water": (0, 255, 0),
    "swimming":     (255, 0, 127),
}

# =============================================================================
# RUNTIME STATE
# =============================================================================

# Frame + predictions — updated by different threads, read by display thread
latest_frame          = None
frame_lock            = threading.Lock()

latest_predictions    = []          # list of dicts: {x,y,width,height,class,confidence}
latest_drown_candidate = False
latest_scene_state    = "UNCERTAIN"
latest_best_confs     = (0.0, 0.0, 0.0)   # (drown, out, swim)
latest_hist_counts    = (0, 0, 0)         # (drown, out, swim)
latest_raw_debug      = "No detections yet"
pred_lock             = threading.Lock()

# JPEG bytes produced by display thread, consumed by Flask generate()
latest_jpeg_bytes     = None
jpeg_event            = threading.Event()
jpeg_lock             = threading.Lock()

# Alarm state
state_history         = deque(maxlen=STATE_HISTORY_LEN)
alarm_active          = False
alarm_start_time      = 0
last_alert_time       = 0
latest_alert          = None

# Pipeline lifecycle (read by /status)
PIPELINE_STATUS = "starting"
PIPELINE_ERROR  = None
_status_lock    = threading.Lock()

# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)

# =============================================================================
# SUPABASE
# =============================================================================

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================================================================
# AUDIO
# =============================================================================

if AUDIO_ENABLED:
    try:
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.mixer.init()
        pygame.mixer.music.load(SIREN_FILE)
        print("Siren loaded")
    except Exception as e:
        AUDIO_ENABLED = False
        print("Audio disabled:", e)


def play_siren():
    if AUDIO_ENABLED:
        try:
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(-1)
        except Exception as e:
            print("Audio play error:", e)


def stop_siren():
    if AUDIO_ENABLED:
        try:
            pygame.mixer.music.stop()
        except Exception as e:
            print("Audio stop error:", e)


# =============================================================================
# FONT
# =============================================================================

try:
    FONT    = ImageFont.truetype(FONT_PATH, 20)
    FONT_SM = ImageFont.truetype(FONT_PATH, 16)
except OSError:
    FONT    = ImageFont.load_default()
    FONT_SM = FONT

# =============================================================================
# HELPERS
# =============================================================================

def resize_for_display(frame: np.ndarray, target_width: int = DISPLAY_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    scale = target_width / w
    return cv2.resize(frame, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)


def log_alert(confidence: float, label: str = "Drowning Detected"):
    global latest_alert
    try:
        data = {
            "camera_id":     "CCTV1",
            "alert_message": label,
            "status":        "ongoing",
            "confidence":    confidence,
            "alert_time":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        supabase.table("alerts").insert(data).execute()
    except Exception as e:
        print("Supabase alert error:", e)

    latest_alert = {
        "message":    label,
        "confidence": round(confidence * 100, 1),
        "timestamp":  time.time(),
    }


# =============================================================================
# DRAWING (PIL-based, matches drowning.py look)
# =============================================================================

def draw_predictions(frame: np.ndarray, predictions: list,
                     drown_candidate: bool) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    for pred in predictions:
        if not isinstance(pred, dict):
            continue

        x, y = pred.get("x", 0), pred.get("y", 0)
        w, h = pred.get("width", 0), pred.get("height", 0)
        cls  = pred.get("class", "")
        conf = pred.get("confidence", 0.0)

        if not DEBUG_PASSTHROUGH:
            # Suppress drowning box unless it's a valid candidate
            if cls == "drowning" and not drown_candidate:
                continue
            # Per-class confidence gates
            if cls == "drowning"     and conf < DROWN_DRAW_THRESHOLD: continue
            if cls == "out of water" and conf < OUT_DRAW_THRESHOLD:   continue
            if cls == "swimming"     and conf < SWIM_DRAW_THRESHOLD:  continue

        color_bgr = CLASS_COLORS_BGR.get(cls, (255, 255, 255))
        color_rgb = color_bgr[::-1]

        pt1 = (int(x - w / 2), int(y - h / 2))
        pt2 = (int(x + w / 2), int(y + h / 2))
        box_w = 8 if cls == "drowning" else 4
        draw.rectangle([pt1, pt2], outline=color_rgb, width=box_w)

        label = f"{cls.upper()} {conf * 100:.1f}%"
        tb = draw.textbbox((0, 0), label, font=FONT)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        lx = pt1[0]
        ly = pt1[1] - th - 4 if pt1[1] - th - 4 >= 0 else pt1[1] + 4
        draw.rectangle([lx, ly, lx + tw + 6, ly + th + 4], fill=color_rgb)
        draw.text((lx + 3, ly + 2), label, font=FONT, fill="white")

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_status_overlay(frame: np.ndarray, scene_state: str,
                        best_drown: float, best_out: float, best_swim: float,
                        drown_count: int, out_count: int, swim_count: int,
                        raw_debug: str) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    if alarm_active:
        status_text, status_color = "DROWNING ALERT!",       (255, 0,   0)
    elif scene_state == "DROWNING":
        status_text, status_color = "POSSIBLE DROWNING",     (255, 60,  0)
    elif scene_state == "OUT":
        status_text, status_color = "PERSON OUT OF WATER",   (255, 200, 0)
    elif scene_state == "SWIMMING":
        status_text, status_color = "SWIMMING DETECTED",     (0,   200, 0)
    else:
        status_text, status_color = "SAFE / UNCERTAIN",      (0,   180, 0)

    # Status badge
    tb = draw.textbbox((0, 0), status_text, font=FONT)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.rectangle([10, 10, 10 + tw + 12, 10 + th + 8], fill=status_color)
    draw.text((16, 14), status_text, font=FONT, fill="white")

    # Confidence + history
    info = (f"D={best_drown:.2f}  O={best_out:.2f}  S={best_swim:.2f}"
            f"    hist D={drown_count} O={out_count} S={swim_count}")
    tb2 = draw.textbbox((0, 0), info, font=FONT_SM)
    tw2, th2 = tb2[2] - tb2[0], tb2[3] - tb2[1]
    y2 = 10 + th + 8 + 6
    draw.rectangle([10, y2, 10 + tw2 + 12, y2 + th2 + 6], fill=(30, 30, 30))
    draw.text((16, y2 + 3), info, font=FONT_SM, fill="white")

    # Raw detections line (stream_bridge.py style — truncated)
    raw = f"Raw: {raw_debug[:110]}"
    tb3 = draw.textbbox((0, 0), raw, font=FONT_SM)
    tw3, th3 = tb3[2] - tb3[0], tb3[3] - tb3[1]
    y3 = y2 + th2 + 6 + 4
    draw.rectangle([10, y3, 10 + tw3 + 12, y3 + th3 + 6], fill=(30, 30, 30))
    draw.text((16, y3 + 3), raw, font=FONT_SM, fill=(200, 200, 255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# =============================================================================
# ALARM LOGIC  (mirrors stream_bridge.py)
# =============================================================================

def _process_alarm(scene_state: str, best_drown_conf: float, drown_count: int):
    global alarm_active, alarm_start_time, last_alert_time

    now = time.time()

    if (
        scene_state == "DROWNING"
        and drown_count >= REQUIRED_DROWN_FRAMES
        and not alarm_active
        and (now - last_alert_time) >= ALERT_COOLDOWN
    ):
        alarm_active     = True
        alarm_start_time = now
        last_alert_time  = now
        play_siren()
        log_alert(best_drown_conf if best_drown_conf > 0 else 0.50)
        print("DROWNING DETECTED")

    if alarm_active:
        if (now - alarm_start_time >= ALARM_HOLD) and scene_state != "DROWNING":
            alarm_active = False
            stop_siren()
            print("Alarm Cleared")


# =============================================================================
# MODEL LOAD
# =============================================================================

print("Loading Roboflow model...")
model = get_model(model_id=RF_MODEL_ID, api_key=RF_API_KEY)
print("Model loaded:", RF_MODEL_ID)


# =============================================================================
# CAPTURE THREAD — owns the video source
# =============================================================================

def capture_loop():
    global latest_frame, PIPELINE_STATUS, PIPELINE_ERROR

    while True:
        src = VIDEO_SOURCE
        is_rtsp = str(src).startswith("rtsp://")

        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if is_rtsp else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            with _status_lock:
                PIPELINE_STATUS = "error"
                PIPELINE_ERROR  = f"Cannot open source: {src}"
            print(f"[CAPTURE] cannot open {src} — retry in {SOURCE_RETRY_DELAY}s")
            time.sleep(SOURCE_RETRY_DELAY)
            continue

        with _status_lock:
            PIPELINE_STATUS = "running"
            PIPELINE_ERROR  = None
        print(f"[CAPTURE] source opened: {src}")

        while True:
            ret, frame = cap.read()
            if not ret:
                if not is_rtsp and LOOP_VIDEO_FILES:
                    # loop the file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                print("[CAPTURE] read failed — reconnecting")
                break

            with frame_lock:
                latest_frame = frame

        cap.release()
        time.sleep(SOURCE_RETRY_DELAY)


# =============================================================================
# DETECT THREAD — runs inference every FRAME_SKIP frames
# =============================================================================

def detect_loop():
    global latest_predictions, latest_drown_candidate, latest_scene_state
    global latest_best_confs, latest_hist_counts, latest_raw_debug

    counter = 0
    print_counter = 0

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        counter += 1
        if counter % FRAME_SKIP != 0:
            time.sleep(0.005)
            continue

        h, w = frame.shape[:2]
        frame_area = max(1, h * w)

        # --- Inference ---
        try:
            results = model.infer(frame, confidence=BASE_CONF)
        except Exception as e:
            print(f"[INFER] error: {e}")
            time.sleep(0.1)
            continue

        raw_preds = []
        if results:
            try:
                raw_preds = [p.model_dump(by_alias=True) for p in results[0].predictions]
            except Exception:
                raw_preds = []

        # --- MIN_AREA_RATIO filter (drop tiny noise boxes) ---
        preds = []
        for p in raw_preds:
            bw, bh = p.get("width", 0), p.get("height", 0)
            if (bw * bh) / frame_area < MIN_AREA_RATIO:
                continue
            preds.append(p)

        # --- Confidence aggregation ---
        best_drown = best_out = best_swim = 0.0
        det_strings = []
        for p in preds:
            cls  = p.get("class", "")
            conf = p.get("confidence", 0.0)
            det_strings.append(f"{cls}:{conf:.2f}")
            if cls == "drowning":
                best_drown = max(best_drown, conf)
            elif cls == "out of water":
                best_out   = max(best_out, conf)
            elif cls == "swimming":
                best_swim  = max(best_swim, conf)

        raw_debug = " | ".join(det_strings[:6]) if det_strings else "No detections"

        # --- Strict drowning suppression (mirrors stream_bridge.py) ---
        drown_candidate = best_drown >= DROWN_ALERT_THRESHOLD
        out_strong  = best_out  >= OUT_DRAW_THRESHOLD
        swim_strong = best_swim >= SWIM_DRAW_THRESHOLD

        if out_strong  and best_out  >= best_drown - HARD_SUPPRESS_MARGIN:
            drown_candidate = False
        if swim_strong and best_swim >= best_drown - HARD_SUPPRESS_MARGIN:
            drown_candidate = False
        if not (
            best_drown >= best_out  + DOMINANCE_MARGIN and
            best_drown >= best_swim + DOMINANCE_MARGIN
        ):
            drown_candidate = False

        # --- Scene state ---
        if drown_candidate:
            scene_state = "DROWNING"
        elif out_strong and best_out >= best_swim:
            scene_state = "OUT"
        elif swim_strong:
            scene_state = "SWIMMING"
        else:
            scene_state = "UNCERTAIN"

        state_history.append(scene_state)
        if scene_state in ("OUT", "SWIMMING"):
            filtered = [s for s in state_history if s != "DROWNING"]
            state_history.clear()
            state_history.extend(filtered)

        drown_count = sum(1 for s in state_history if s == "DROWNING")
        out_count   = sum(1 for s in state_history if s == "OUT")
        swim_count  = sum(1 for s in state_history if s == "SWIMMING")

        # --- Alarm ---
        _process_alarm(scene_state, best_drown, drown_count)

        # --- Publish to display thread ---
        with pred_lock:
            latest_predictions     = preds
            latest_drown_candidate = drown_candidate
            latest_scene_state     = scene_state
            latest_best_confs      = (best_drown, best_out, best_swim)
            latest_hist_counts     = (drown_count, out_count, swim_count)
            latest_raw_debug       = raw_debug

        # --- Debug print ---
        if DEBUG_PRINT_EVERY > 0:
            print_counter += 1
            if print_counter >= DEBUG_PRINT_EVERY:
                print_counter = 0
                print(f"[INFER] D={best_drown:.2f} O={best_out:.2f} S={best_swim:.2f} "
                      f"state={scene_state} cand={drown_candidate} raw=[{raw_debug}]")


# =============================================================================
# DISPLAY / ENCODE THREAD — draws cached boxes on every fresh frame
# =============================================================================

def display_loop():
    global latest_jpeg_bytes

    frame_interval = 1.0 / max(1, TARGET_DISPLAY_FPS)
    next_tick = time.time()

    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            next_tick = time.time()
            continue

        with pred_lock:
            preds       = list(latest_predictions)
            drown_cand  = latest_drown_candidate
            ss          = latest_scene_state
            bd, bo, bs  = latest_best_confs
            dc, oc, sc  = latest_hist_counts
            raw_debug   = latest_raw_debug

        display = draw_predictions(frame, preds, drown_cand)
        display = draw_status_overlay(display, ss, bd, bo, bs, dc, oc, sc, raw_debug)
        display = resize_for_display(display, DISPLAY_WIDTH)

        ok, buf = cv2.imencode(".jpg", display,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue

        with jpeg_lock:
            latest_jpeg_bytes = buf.tobytes()
        jpeg_event.set()

        # Pace to target display fps
        next_tick += frame_interval
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()


# =============================================================================
# FLASK — MJPEG STREAM (event-driven, no re-encode loop)
# =============================================================================

def generate():
    last_bytes = None
    while True:
        # Wait until display thread pushes a new frame, or timeout heartbeat
        got = jpeg_event.wait(timeout=1.0)
        if got:
            jpeg_event.clear()
        with jpeg_lock:
            b = latest_jpeg_bytes
        if b is None or b is last_bytes:
            if not got:
                continue
        last_bytes = b
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + b
            + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/latest_alert")
def latest_alert_api():
    return jsonify(latest_alert or {
        "message": None, "confidence": None, "timestamp": None
    })


@app.route("/status")
def status_api():
    with _status_lock:
        return jsonify({
            "pipeline":        PIPELINE_STATUS,
            "error":           PIPELINE_ERROR,
            "source":          str(VIDEO_SOURCE),
            "frame_skip":      FRAME_SKIP,
            "display_fps":     TARGET_DISPLAY_FPS,
            "display_width":   DISPLAY_WIDTH,
            "jpeg_quality":    JPEG_QUALITY,
            "debug_passthrough": DEBUG_PASSTHROUGH,
        })


# --- Camera management (same as stream_bridge.py) ---

@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    try:
        res = supabase.table("cameras").select("*").execute()
        return jsonify(res.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cameras", methods=["POST"])
def add_camera():
    try:
        data = request.json
        if not data or "id" not in data or "rtsp_url" not in data:
            return jsonify({"error": "Missing required fields"}), 400
        cam_data = {
            "id":        data["id"],
            "rtsp_url":  data["rtsp_url"],
            "is_active": data.get("is_active", True),
        }
        res = supabase.table("cameras").upsert(cam_data, on_conflict=["id"]).execute()
        return jsonify(res.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def delete_camera(cam_id):
    try:
        res = supabase.table("cameras").delete().eq("id", cam_id).execute()
        return jsonify(res.data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# MAIN
# =============================================================================

def register_camera():
    try:
        supabase.table("cameras").upsert(
            {"id": "CCTV1", "rtsp_url": str(VIDEO_SOURCE), "is_active": True},
            on_conflict=["id"],
        ).execute()
        print("Camera registered in Supabase")
    except Exception as e:
        print("Camera registration error:", e)


if __name__ == "__main__":
    register_camera()

    def _shutdown(sig, frame):
        print("\n[SHUTDOWN] Stopping...")
        with _status_lock:
            global PIPELINE_STATUS
            PIPELINE_STATUS = "stopped"
        stop_siren()
        os._exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=detect_loop,  daemon=True).start()
    threading.Thread(target=display_loop, daemon=True).start()

    print("[SERVER] Running")
    print("  Stream → http://localhost:5001/video_feed")
    print("  Alert  → http://localhost:5001/latest_alert")
    print("  Status → http://localhost:5001/status")
    print(f"  FRAME_SKIP={FRAME_SKIP}  TARGET_DISPLAY_FPS={TARGET_DISPLAY_FPS}  "
          f"DEBUG_PASSTHROUGH={DEBUG_PASSTHROUGH}")
    app.run(host="0.0.0.0", port=5001, threaded=True, use_reloader=False)
