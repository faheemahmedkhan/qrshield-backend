# main.py
import io
import os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pyzbar.pyzbar import decode as pyzbar_decode
import cv2

import qrdl
import qrml

app = FastAPI(title="QRShield API")

# Mount the static directory so we can serve Logo.png
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

DEBUG     = False
DEBUG_DIR = "qr_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

import gc

print("Loading DL model...")
dl_model = qrdl.load_model()
print("DL model loaded.")
gc.collect() # Force garbage collection of massive setup artifacts

# ──────────────────────────────────────────
# Debug helper
# ──────────────────────────────────────────
def save_debug(img_pil, name):
    if not DEBUG:
        return
    path = os.path.join(DEBUG_DIR, f"{int(time.time()*1000)}_{name}.png")
    try:
        img_pil.save(path)
    except Exception as e:
        print("[DEBUG] save failed:", e)

# ──────────────────────────────────────────
# Low-level decoders
# ──────────────────────────────────────────
def try_pyzbar(pil_image):
    try:
        decoded = pyzbar_decode(pil_image)
        for obj in decoded:
            try:    return obj.data.decode("utf-8")
            except: return obj.data.decode(errors="ignore")
    except Exception:
        pass
    return None

def try_opencv_qr(pil_image):
    try:
        arr      = np.array(pil_image.convert("RGB"))[:, :, ::-1]
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(arr)
        if data and data.strip():
            return data
    except Exception:
        pass
    return None

# ──────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────
def upscale_image(img, scale):
    w, h = img.size
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def adaptive_threshold_cv(img):
    arr = np.array(img.convert("L"))
    try:
        thr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 10)
        return Image.fromarray(thr)
    except Exception:
        return img

def denoise_cv(img):
    arr = np.array(img.convert("RGB"))
    try:
        dst = cv2.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)
        return Image.fromarray(dst)
    except Exception:
        return img

def gamma_correction(img, gamma):
    lut = np.array([pow(x / 255.0, 1.0 / gamma) * 255 for x in range(256)]).clip(0, 255).astype("uint8")
    return img.point(lambda i: lut[i])

# ──────────────────────────────────────────
# Robust decode
# ──────────────────────────────────────────
def robust_decode(pil_image):
    save_debug(pil_image, "orig")
    r = try_pyzbar(pil_image);   
    if r: return r, "pyzbar:orig"
    r = try_opencv_qr(pil_image)
    if r: return r, "opencv:orig"

    scales    = [1, 1.5, 2, 3]
    rotations = [0, 90, 180, 270]
    pre_funcs = [
        lambda x: x,
        lambda x: ImageOps.autocontrast(x),
        lambda x: x.filter(ImageFilter.UnsharpMask(1, 150, 3)),
        lambda x: denoise_cv(ImageOps.autocontrast(x)),
        lambda x: adaptive_threshold_cv(ImageOps.autocontrast(x)),
        lambda x: gamma_correction(ImageOps.autocontrast(x), 0.8),
        lambda x: gamma_correction(ImageOps.autocontrast(x), 1.2),
    ]

    for scale in scales:
        scaled = pil_image if scale == 1 else upscale_image(pil_image, scale)
        for rot in rotations:
            cand = scaled.rotate(rot, expand=True) if rot != 0 else scaled
            for pf in pre_funcs:
                ti = pf(cand)
                r  = try_pyzbar(ti)
                if r: return r, f"pyzbar:s{scale}_r{rot}"
                r  = try_opencv_qr(ti)
                if r: return r, f"opencv:s{scale}_r{rot}"

    # Last-resort Otsu
    try:
        arr = np.array(pil_image.convert("L"))
        for k in [3, 5, 7]:
            b     = cv2.medianBlur(arr, k)
            _, thr = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            timg  = Image.fromarray(thr)
            r = try_pyzbar(timg)
            if r: return r, f"pyzbar:otsu_k{k}"
            r = try_opencv_qr(timg)
            if r: return r, f"opencv:otsu_k{k}"
    except Exception:
        pass

    return None, "not_decoded"

# ──────────────────────────────────────────
# Fusion & decision
# ──────────────────────────────────────────
def adaptive_fusion(ml_prob, dl_prob):
    if ml_prob < 0.30:
        w_ml, w_dl, mode = 0.95, 0.05, "ML Dominant (Benign)"
    elif ml_prob < 0.70:
        w_ml, w_dl, mode = 0.80, 0.20, "Balanced (Suspicious)"
    else:
        w_ml, w_dl, mode = 0.95, 0.05, "ML Dominant (Malicious)"
    return (w_ml * ml_prob) + (w_dl * dl_prob), mode

def risk_decision(score):
    if score < 0.25:  return "SAFE"
    if score < 0.60:  return "SUSPICIOUS"
    return "MALICIOUS"

# ──────────────────────────────────────────
# Shared result skeleton
# ──────────────────────────────────────────
def _empty_result():
    return {
        "decoded_url":      None,
        "is_short_url":     False,
        "ml_probability":   0.0,
        "dl_probability":   0.0,
        "fusion_score":     0.0,
        "fusion_mode":      "",
        "status":           "UNKNOWN",
        "error":            None,
        "note":             None,
        "shap_explanation": [],
    }

# ──────────────────────────────────────────
# Main QR pipeline
# ──────────────────────────────────────────
def analyze_qr(image):
    result = _empty_result()
    try:
        image     = image.convert("RGB")
        temp_path = "temp_scan.png"
        image.save(temp_path, format="PNG")

        # DL model always runs on the image
        _, dl_prob = qrdl.predict_image(dl_model, temp_path)
        result["dl_probability"] = round(float(dl_prob), 6)

        decoded, method = robust_decode(image)
        if not decoded:
            result["fusion_score"] = round(float(dl_prob), 6)
            result["fusion_mode"]  = "DL Only (No URL)"
            result["status"]       = "DISTORTED_QR" if dl_prob > 0 else "UNKNOWN"
            result["note"]         = "No URL decoded after all preprocessing attempts."
            return result

        result["decoded_url"] = decoded
        result["note"]        = f"decoded_by={method}"

        # ── Short URL path ──────────────────────────
        if qrml.is_short_url(decoded):
            result["is_short_url"]   = True
            result["ml_probability"] = 0.0
            result["fusion_score"]   = round(float(dl_prob), 6)
            result["fusion_mode"]    = "Short URL — ML skipped"
            result["status"]         = "SUSPICIOUS"
            # SHAP not run; destination is unknown
            return result

        # ── Normal ML path ──────────────────────────
        url = decoded if decoded.startswith(("http://", "https://")) else "https://" + decoded
        try:
            _, ml_prob, shap_explanation = qrml.predict_url(url)
            result["ml_probability"]   = round(float(ml_prob), 6)
            result["shap_explanation"] = shap_explanation
        except Exception as e:
            result["ml_probability"] = 0.0
            result["note"]          += f"; ml_error={e}"

        fusion_score, fusion_mode = adaptive_fusion(result["ml_probability"], dl_prob)
        result["fusion_score"] = round(float(fusion_score), 6)
        result["fusion_mode"]  = fusion_mode
        result["status"]       = risk_decision(fusion_score)

    except Exception as e:
        result["error"] = str(e)

    return result

# ──────────────────────────────────────────
# Manual URL analysis (for short URL follow-up)
# ──────────────────────────────────────────
class UrlPayload(BaseModel):
    url: str

def analyze_url_directly(url: str):
    """Run ML + SHAP on a URL string (no QR image, no DL model)."""
    result = _empty_result()
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        result["decoded_url"] = url
        _, ml_prob, shap_explanation = qrml.predict_url(url)
        result["ml_probability"]   = round(float(ml_prob), 6)
        result["dl_probability"]   = 0.0          # no image — DL not applicable
        result["shap_explanation"] = shap_explanation
        # Fusion with DL=0 and full ML weight
        result["fusion_score"]  = round(float(ml_prob), 6)
        result["fusion_mode"]   = "ML Only (Manual URL)"
        result["status"]        = risk_decision(ml_prob)
    except Exception as e:
        result["error"] = str(e)
    return result

# ──────────────────────────────────────────
# FastAPI routes
# ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/scan")
async def scan_qr(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
        return analyze_qr(image)
    except Exception as e:
        r         = _empty_result()
        r["error"]  = str(e)
        r["status"] = "ERROR"
        r["note"]   = "Failed to process uploaded image."
        return r

@app.post("/analyze-url")
async def analyze_url(payload: UrlPayload):
    """Accepts a raw URL string and returns ML analysis + SHAP."""
    try:
        return analyze_url_directly(payload.url.strip())
    except Exception as e:
        r         = _empty_result()
        r["error"]  = str(e)
        r["status"] = "ERROR"
        return r
