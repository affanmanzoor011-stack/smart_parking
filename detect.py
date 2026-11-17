"""
detect.py  -  Day 1 module (v2: better plate reading)
Detects vehicles (YOLOv8) and reads the number plate (EasyOCR).

Functions other modules will use:
    detect_vehicles(image_path) -> list of dicts
    detect_plate(image_path)    -> plate text (str)

Test from terminal:
    python detect.py test_images/car1.jpg
"""

import re
import sys

import cv2
import easyocr
from ultralytics import YOLO

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Indian plate: 2 letters, 1-2 digits, 1-3 letters, 4 digits
# e.g. HR55ET7097, DL8CAZ9348
SEARCH_RE = re.compile(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}")
FULL_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "UA", "WB",
}

# Common OCR mix-ups
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "S": "5", "B": "8", "G": "6"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"}

# Models load once (first run downloads them, needs internet)
model = YOLO("yolov8n.pt")
reader = easyocr.Reader(["en"], gpu=False)


# ---------------------------------------------------------------- vehicles
def _overlap_small(a, b):
    """Intersection area divided by the smaller box's area."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller else 0


def detect_vehicles(image_path, min_conf=0.4):
    """Return vehicles found: type, confidence, box (x1, y1, x2, y2)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    result = model(img, verbose=False)[0]
    found = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls in VEHICLE_CLASSES and conf >= min_conf:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            found.append({"type": VEHICLE_CLASSES[cls],
                          "confidence": round(conf, 2),
                          "box": (x1, y1, x2, y2)})

    # Remove duplicate boxes on the same vehicle (keep higher confidence)
    found.sort(key=lambda v: v["confidence"], reverse=True)
    kept = []
    for v in found:
        if all(_overlap_small(v["box"], k["box"]) < 0.7 for k in kept):
            kept.append(v)
    return kept


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
    """Upscale small images, then run OCR. Returns (bbox, text, conf) list."""
    h, w = img.shape[:2]
    if w < 800:
        scale = 800 / w
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)
    return reader.readtext(img)


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
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    vehicles = detect_vehicles(image_path)
    vehicles.sort(key=lambda v: (v["box"][2] - v["box"][0]) *
                  (v["box"][3] - v["box"][1]), reverse=True)

    all_cands = []
    fallbacks = []

    # 1) Read each vehicle (largest first, max 4)
    for v in vehicles[:4]:
        x1, y1, x2, y2 = v["box"]
        crop = img[max(y1, 0):y2, max(x1, 0):x2]
        if crop.size == 0:
            continue
        c, fb = _candidates(_read(crop))
        all_cands += c
        fallbacks.append(fb)

    # 2) If nothing valid, read the whole image, then a contrast-boosted copy
    if not all_cands:
        c, fb = _candidates(_read(img))
        all_cands += c
        fallbacks.append(fb)
    if not all_cands:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        c, fb = _candidates(_read(gray))
        all_cands += c
        fallbacks.append(fb)

    if all_cands:
        return max(all_cands, key=lambda x: x[1])[0]

    # Nothing valid: return the most confident raw text (user can edit it)
    best = max(fallbacks, key=lambda x: x[1]) if fallbacks else ("", 0)
    return _finalize(best[0]) if best[1] > 0.3 else ""


# --------------------------------------------------------------- annotated
def save_annotated(image_path, out_path="output.jpg"):
    """Draw boxes on the image and save it."""
    img = cv2.imread(image_path)
    for v in detect_vehicles(image_path):
        x1, y1, x2, y2 = v["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{v["type"]} {v["confidence"]}',
                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, img)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    print("Vehicles:", detect_vehicles(path))
    print("Plate   :", detect_plate(path))
    print("Saved   :", save_annotated(path))
