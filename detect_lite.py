"""
detect_lite.py  -  low-memory detection (ONNX Runtime, no PyTorch)

No PyTorch. Uses:
  - onnxruntime + yolov8n.onnx   for vehicle detection
  - RapidOCR (onnxruntime)       for plate reading

Memory tricks:
  - only ONE model is in memory at a time (YOLO is freed before OCR loads,
    OCR is freed after each photo)
  - huge JPEG photos are decoded at reduced size
  - single-thread inference, no memory arena

Functions:
    detect_vehicles(image_path) -> list of dicts
    detect_plate(image_path)    -> plate text (str)
    save_annotated(image_path, out_path) -> path

Switches (environment variables):
    YOLO_ONNX=yolov8n.onnx          path of the ONNX model
    SKIP_VEHICLE_DETECTION=1        skip YOLO completely (plate-only, saves more memory)

Test from terminal:
    python detect_lite.py "test images/car1.jpg"
"""

import gc
import os
import re
import sys

import cv2
import numpy as np

MODEL_PATH = os.environ.get("YOLO_ONNX", "yolov8n.onnx")
SKIP_VEHICLE = os.environ.get("SKIP_VEHICLE_DETECTION") == "1"
MAX_SIDE = 1280  # big phone photos are shrunk to this (saves RAM and time)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Indian plate: 2 letters, 1-2 digits, 1-3 letters, 4 digits
SEARCH_RE = re.compile(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}")
FULL_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "UA", "WB",
}

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}


# ------------------------------------------------------- lazy model loading
_yolo = None
_ocr = None


def _get_yolo():
    global _yolo
    if _yolo is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        _yolo = ort.InferenceSession(
            MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
    return _yolo


def _get_ocr():
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr = RapidOCR()
    return _ocr


def _free_memory():
    gc.collect()
    try:  # Linux only: hand freed memory back to the system
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _release_yolo():
    global _yolo
    _yolo = None
    _free_memory()


def _release_ocr():
    global _ocr
    _ocr = None
    _free_memory()


# ------------------------------------------------------------------ helpers
def _load(image_path):
    """Read an image (huge JPEGs are decoded smaller) and shrink it if needed."""
    img = None
    try:
        from PIL import Image, ImageOps

        with Image.open(image_path) as im:
            im.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG only: decode at lower size
            im = ImageOps.exif_transpose(im)       # phone photos: fix rotation
            img = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > MAX_SIDE:
        s = MAX_SIDE / longest
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _overlap_small(a, b):
    """Intersection area divided by the smaller box's area."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller else 0


# ---------------------------------------------------------------- vehicles
def _preprocess(img, size):
    """Letterbox to size x size and convert to a float32 NCHW blob."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), r, left, top


def _postprocess(out, r, left, top, width, height, min_conf):
    """Turn raw YOLOv8 output into vehicle boxes in original-image pixels."""
    preds = out[0]
    if preds.shape[0] < preds.shape[1]:  # (84, 8400) -> (8400, 84)
        preds = preds.T

    scores = preds[:, 4:]
    ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)
    keep = np.isin(ids, list(VEHICLE_CLASSES)) & (confs >= min_conf)
    if not keep.any():
        return []

    boxes, confs, ids = preds[keep, :4], confs[keep], ids[keep]
    x1 = (boxes[:, 0] - boxes[:, 2] / 2 - left) / r
    y1 = (boxes[:, 1] - boxes[:, 3] / 2 - top) / r
    ws = boxes[:, 2] / r
    hs = boxes[:, 3] / r
    rects = [[int(a), int(b), int(c), int(d)] for a, b, c, d in zip(x1, y1, ws, hs)]

    idx = cv2.dnn.NMSBoxes(rects, confs.astype(float).tolist(), min_conf, 0.5)
    idx = np.array(idx).flatten() if len(idx) else []

    found = []
    for i in idx:
        x, y, w, h = rects[i]
        bx1, by1 = max(0, x), max(0, y)
        bx2, by2 = min(width, x + w), min(height, y + h)
        if bx2 > bx1 and by2 > by1:
            found.append({"type": VEHICLE_CLASSES[int(ids[i])],
                          "confidence": round(float(confs[i]), 2),
                          "box": (bx1, by1, bx2, by2)})

    # Remove duplicate boxes on the same vehicle (keep higher confidence)
    found.sort(key=lambda v: v["confidence"], reverse=True)
    kept = []
    for v in found:
        if all(_overlap_small(v["box"], k["box"]) < 0.7 for k in kept):
            kept.append(v)
    return kept


def _detect_on(img, min_conf=0.4):
    sess = _get_yolo()
    shape = sess.get_inputs()[0].shape
    size = shape[2] if isinstance(shape[2], int) else 640
    blob, r, left, top = _preprocess(img, size)
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    h, w = img.shape[:2]
    return _postprocess(out, r, left, top, w, h, min_conf)


# Remember the last image's result so YOLO runs only once per upload
_cache = {}


def _vehicles_for(image_path, img):
    if SKIP_VEHICLE:
        return []
    if image_path not in _cache:
        _cache.clear()
        try:
            _cache[image_path] = _detect_on(img)
        finally:
            _release_yolo()  # free YOLO before OCR gets loaded
    return list(_cache[image_path])


def detect_vehicles(image_path, min_conf=0.4):
    """Return vehicles found: type, confidence, box (x1, y1, x2, y2)."""
    img = _load(image_path)
    return _vehicles_for(image_path, img)


# ------------------------------------------------------------------ plates
def _clean(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _fix_format(s):
    """Try to correct OCR mix-ups so s matches the plate format."""
    n = len(s)
    if n < 8 or n > 11:
        return None
    for d in (2, 1):  # RTO code can have 2 digits (UP65) or 1 digit (DL8)
        letters = n - 4 - 2 - d
        if not 1 <= letters <= 3:
            continue
        c = list(s)
        for i in (0, 1):
            c[i] = TO_LETTER.get(c[i], c[i])
        for i in range(2, 2 + d):
            c[i] = TO_DIGIT.get(c[i], c[i])
        for i in range(2 + d, n - 4):
            c[i] = TO_LETTER.get(c[i], c[i])
        for i in range(n - 4, n):
            c[i] = TO_DIGIT.get(c[i], c[i])
        out = "".join(c)
        if FULL_RE.match(out):
            return out
    return None


def _read(img):
    """Run OCR. Returns a list of (bbox, text, conf)."""
    h, w = img.shape[:2]
    scale = 1.0
    if w < 500:  # enlarge very small crops
        scale = 500 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    result, _ = _get_ocr()(img, use_cls=False)
    items = []
    for box, text, conf in (result or []):
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        # map the box back to the coordinates of the image that was passed in
        box = [[p[0] / scale, p[1] / scale] for p in box]
        items.append((box, text, conf))
    return items


def _variants(img):
    """Same image in 3 versions (plain, contrast-boosted, sharpened)."""
    yield img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    yield cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    yield cv2.addWeighted(img, 1.6, blur, -0.6, 0)


def _plate_crop(img, results, target_w=700):
    """
    Use the first OCR pass to find where the plate is, cut a tight crop
    around it and enlarge it. Returns None if no plate-like text was found.
    """
    boxes = []
    for bbox, text, conf in results:
        t = _clean(text)
        digits = sum(ch.isdigit() for ch in t)
        if len(t) >= 3 and digits >= 2:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            boxes.append((min(xs), min(ys), max(xs), max(ys), digits))
    if not boxes:
        return None

    # anchor = the text piece with the most digits (usually the 4 plate digits)
    ax1, ay1, ax2, ay2, _ = max(boxes, key=lambda b: b[4])
    aw, ah = ax2 - ax1, ay2 - ay1

    # keep neighbours that are close to the anchor (other plate line)
    keep = [(ax1, ay1, ax2, ay2)]
    for x1, y1, x2, y2, _ in boxes:
        if (x1, y1, x2, y2) == (ax1, ay1, ax2, ay2):
            continue
        close_y = abs((y1 + y2) / 2 - (ay1 + ay2) / 2) <= ah * 3
        close_x = x2 >= ax1 - aw * 0.6 and x1 <= ax2 + aw * 0.6
        if close_y and close_x:
            keep.append((x1, y1, x2, y2))

    x1 = min(b[0] for b in keep)
    y1 = min(b[1] for b in keep)
    x2 = max(b[2] for b in keep)
    y2 = max(b[3] for b in keep)
    mx, my = (x2 - x1) * 0.12, (y2 - y1) * 0.25
    H, W = img.shape[:2]
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
    if x2 - x1 < 20 or y2 - y1 < 10:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.shape[1] < target_w:
        s = target_w / crop.shape[1]
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return crop


def _vote(runs):
    """
    runs = one candidate list per OCR run. A plate that shows up in several
    runs (different crops / image versions) wins over a one-off misreading.
    """
    score = {}
    for cands in runs:
        best = {}
        for plate, s in cands:
            best[plate] = max(best.get(plate, 0), s)
        for plate, s in best.items():
            n, tot = score.get(plate, (0, 0.0))
            score[plate] = (n + 1, tot + s)
    if not score:
        return None
    return max(score.items(), key=lambda kv: kv[1])[0]


def _candidates(results):
    """Turn OCR results into (plate, score) candidates and a raw fallback."""
    items = []
    for bbox, text, conf in results:
        t = _clean(text)
        if not t:
            continue
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        items.append({"t": t, "conf": float(conf), "x": min(xs),
                      "yc": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys)})
    items.sort(key=lambda i: (i["yc"], i["x"]))

    # Group text pieces that sit on the same line
    lines = []
    for it in items:
        for line in lines:
            if abs(line[0]["yc"] - it["yc"]) < max(line[0]["h"], it["h"]) * 0.6:
                line.append(it)
                break
        else:
            lines.append([it])

    line_texts = []
    for line in lines:
        line.sort(key=lambda i: i["x"])
        text = "".join(i["t"] for i in line)
        conf = sum(i["conf"] for i in line) / len(line)
        line_texts.append((text, conf))

    combos = list(line_texts)
    # two-line plates (common on bikes): join consecutive lines
    for a, b in zip(line_texts, line_texts[1:]):
        combos.append((a[0] + b[0], (a[1] + b[1]) / 2))

    cands = []
    for text, conf in combos:
        for m in SEARCH_RE.finditer(text):
            if m.group()[:2] in STATE_CODES:
                cands.append((m.group(), conf + 1))
        for length in (10, 9, 11, 8):
            for s in range(len(text) - length + 1):
                fixed = _fix_format(text[s:s + length])
                if fixed and fixed[:2] in STATE_CODES:
                    cands.append((fixed, conf))

    fallback = max(line_texts, key=lambda x: x[1]) if line_texts else ("", 0)
    return cands, fallback


def _finalize(text):
    """Last clean-up: pull out a valid plate, or drop the 'IND' strip text."""
    text = _clean(text)
    for m in SEARCH_RE.finditer(text):
        if m.group()[:2] in STATE_CODES:
            return m.group()
    if text.startswith("IND"):
        text = text[3:]
    return text


def detect_plate(image_path):
    """Return the plate text. Best guess (or empty) if no valid plate found."""
    try:
        return _detect_plate(image_path)
    finally:
        _release_ocr()  # free the OCR model after every photo


def _detect_plate(image_path):
    img = _load(image_path)
    vehicles = _vehicles_for(image_path, img)
    vehicles.sort(key=lambda v: (v["box"][2] - v["box"][0]) *
                  (v["box"][3] - v["box"][1]), reverse=True)

    runs = []        # one candidate list per OCR run
    fallbacks = []

    def read_area(area):
        """OCR an area, then re-read a tight, enlarged crop of the plate."""
        res = _read(area)
        c, fb = _candidates(res)
        runs.append(c)
        fallbacks.append(fb)
        plate_img = _plate_crop(area, res)
        if plate_img is not None:
            for variant in _variants(plate_img):
                c2, fb2 = _candidates(_read(variant))
                runs.append(c2)
                fallbacks.append(fb2)

    # 1) Read the largest vehicles (max 2, to keep the free CPU fast enough)
    for v in vehicles[:2]:
        x1, y1, x2, y2 = v["box"]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        read_area(crop)

    # 2) If nothing valid, read the whole image
    if not any(runs):
        read_area(img)

    winner = _vote(runs)
    if winner:
        return winner

    best = max(fallbacks, key=lambda x: x[1]) if fallbacks else ("", 0)
    return _finalize(best[0]) if best[1] > 0.3 else ""


# --------------------------------------------------------------- annotated
def save_annotated(image_path, out_path="output.jpg"):
    """Draw boxes on the image and save it."""
    img = _load(image_path)
    for v in _vehicles_for(image_path, img):
        x1, y1, x2, y2 = v["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{v["type"]} {v["confidence"]}',
                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_lite.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    print("Vehicles:", detect_vehicles(path))
    print("Plate   :", detect_plate(path))
    print("Saved   :", save_annotated(path))
