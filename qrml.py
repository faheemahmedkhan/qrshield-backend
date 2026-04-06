import cv2
from pyzbar.pyzbar import decode
import joblib
import pandas as pd
import re
import requests
import dns.resolver
import tldextract
from urllib.parse import urlparse
import os
import shap

# ======================================
# CONFIGURE YOUR API
# ======================================

RAPIDAPI_KEY      = "72db2fc05emsh590e54d55c9d522p14c2a1jsn11950d2e12fe"
RAPIDAPI_HOST     = "whois-lookup-api.p.rapidapi.com"
RAPIDAPI_ENDPOINT = "https://whois-lookup-api.p.rapidapi.com/domains-age"

# ======================================
# KNOWN SHORT URL DOMAINS
# ======================================

SHORT_URL_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "short.io", "bl.ink",
    "shorturl.at", "tiny.cc", "urlz.fr", "clck.ru", "bc.vc",
    "adf.ly", "link.tl", "za.gl", "s.id", "rb.gy",
}

# ======================================
# LOAD MODEL
# ======================================

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "xgboost_qr_detector.pkl")
FEATURE_PATH  = os.path.join(BASE_DIR, "feature_columns.pkl")

model           = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

try:
    _shap_explainer = shap.TreeExplainer(model)
except Exception:
    _shap_explainer = None

WHOIS_FALLBACK_AGE_DAYS = 90
WHOIS_MAX_AGE_DAYS      = 3650



# ======================================
# SHORT URL DETECTION
# ======================================

def is_short_url(url):
    """Return True if the URL belongs to a known shortener domain."""
    try:
        ext    = tldextract.extract(url)
        domain = f"{ext.domain}.{ext.suffix}".lower()
        return domain in SHORT_URL_DOMAINS
    except Exception:
        return False

# ======================================
# WHOIS
# ======================================

def get_domain_age(domain):
    headers = {
        "X-RapidAPI-Key":  RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    try:
        resp = requests.get(
            RAPIDAPI_ENDPOINT,
            headers=headers,
            params={"domains": domain},
            timeout=10,
        )
        data = resp.json()
        if domain in data:
            age = data[domain].get("age_days")
            if age is not None:
                return max(0, min(int(age), WHOIS_MAX_AGE_DAYS)), 1
    except Exception:
        pass
    return WHOIS_FALLBACK_AGE_DAYS, 0

# ======================================
# FEATURE EXTRACTION
# ======================================

def extract_features(url):
    features = {}
    parsed   = urlparse(url)
    ext      = tldextract.extract(url)
    domain   = ext.top_domain_under_public_suffix

    # Lexical
    features["checking_ip_address"]  = 1 if re.match(r"^\d+\.\d+\.\d+\.\d+", parsed.netloc) else 0
    features["abnormal_url"]         = 0 if parsed.netloc in url else 1
    features["count_dot"]            = url.count(".")
    features["count_at"]             = url.count("@")
    features["find_dir"]             = url.count("/")
    features["no_of_embed"]          = url.count("//") - 1
    features["shortening_service"]   = 1 if any(s in url for s in [
        "bit.ly","tinyurl.com","goo.gl","t.co","ow.ly","is.gd","buff.ly"
    ]) else 0
    features["count_per"]            = url.count("%")
    features["count_ques"]           = url.count("?")
    features["count_dash"]           = url.count("-")
    features["count_equal"]          = url.count("=")
    features["url_length"]           = len(url)
    features["hostname_length"]      = len(parsed.netloc)
    features["suspicious_words"]     = 1 if any(w in url.lower() for w in [
        "login","verify","update","account","secure","bank","confirm"
    ]) else 0
    features["digit_count"]          = sum(c.isdigit() for c in url)
    features["count_special_chars"]  = len(re.findall(r"[^\w]", url))
    features["fd_length"]            = len(parsed.path.split("/")[1]) if len(parsed.path.split("/")) > 1 else 0
    features["tld_length"]           = len(ext.suffix)
    features["uses_https"]           = 1 if parsed.scheme == "https" else 0

    # WHOIS
    age, available = get_domain_age(domain)
    features["domain_age_days"]  = age
    features["whois_available"]  = available

    # DNS
    try:
        answers = dns.resolver.resolve(domain, "A")
        features["dns_resolves"]      = 1
        features["num_ip_addresses"]  = len(answers)
    except Exception:
        features["dns_resolves"]      = 0
        features["num_ip_addresses"]  = 0

    try:
        dns.resolver.resolve(domain, "MX")
        features["has_mx_record"] = 1
    except Exception:
        features["has_mx_record"] = 0

    # HTTP
    try:
        resp = requests.get(url, timeout=5, allow_redirects=True)
        features["http_status_code"] = resp.status_code
        features["redirect_count"]   = len(resp.history)
        features["ssl_valid"]        = 1 if url.startswith("https") else 0
    except Exception:
        features["http_status_code"] = -1
        features["redirect_count"]   = 0
        features["ssl_valid"]        = 0

    return features

# ======================================
# SHAP EXPLAINABILITY
# ======================================

def get_shap_explanation(df_row):
    """Return [{feature, value, shap_value}, ...] sorted by |shap_value| desc."""
    if _shap_explainer is None:
        return []
    try:
        shap_values = _shap_explainer.shap_values(df_row)
        sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        explanations = [
            {
                "feature":    fname,
                "value":      round(float(fval), 4),
                "shap_value": round(float(sval), 6),
            }
            for fname, fval, sval in zip(df_row.columns.tolist(), df_row.values[0], sv)
        ]
        explanations.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return explanations
    except Exception as e:
        print(f"[SHAP] failed: {e}")
        return []

# ======================================
# PREDICTION  — returns (prediction, prob, shap_explanation)
# ======================================

def predict_url(url):
    """
    Analyse a URL and return:
        prediction      : 0 (benign) or 1 (malicious)
        prob            : float 0-1 malicious probability
        shap_explanation: list of feature contribution dicts
    """
    features = extract_features(url)
    df       = pd.DataFrame([features])
    df       = df.reindex(columns=feature_columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]

    # Compensation rules
    if features.get("whois_available", 0) == 0:
        prob = max(0.0, prob - 0.05)
    if features.get("domain_age_days", 0) > 365 and features.get("suspicious_words", 0) == 0:
        prob = max(0.0, prob - 0.03)
    if features.get("domain_age_days", 0) < 30 and features.get("suspicious_words", 0) == 1:
        prob = min(1.0, prob + 0.08)

    prediction       = 1 if prob >= 0.70 else 0
    shap_explanation = get_shap_explanation(df)

    return prediction, prob, shap_explanation


