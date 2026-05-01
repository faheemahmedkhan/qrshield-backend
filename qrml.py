import cv2 # trigger sync 2
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

_shap_init_error = "None"
try:
    try:
        _shap_explainer = shap.TreeExplainer(model.get_booster())
    except Exception:
        try:
            _shap_explainer = shap.TreeExplainer(model)
        except Exception:
            import numpy as np
            import pandas as pd
            
            # Wrapper to safely dodge scikit-learn read-only attributes on the class instance
            def proxy_predict(X):
                return model.predict_proba(pd.DataFrame(X, columns=feature_columns))

            bg = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
            _shap_explainer = shap.KernelExplainer(proxy_predict, bg)

except Exception as e:
    import traceback
    traceback.print_exc()
    _shap_init_error = str(e)
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
        return [{"feature": f"Init Error: {_shap_init_error}", "value": 0.0, "shap_value": 0.0}]
    try:
        shap_values = _shap_explainer.shap_values(df_row.values)
        
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            if len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]
                
        # FORCE DEBUG: 
        if len(sv) == 0 or len(sv) != len(df_row.columns):
            return [{"feature": f"SV Shape Error. sv len: {len(sv)}, cols: {len(df_row.columns)}", "value": 0.0, "shap_value": 0.0}]
                
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
        import traceback
        traceback.print_exc()
        return [{"feature": f"SHAP Error: {str(e)}", "value": 0.0, "shap_value": 0.0}]

# ======================================
# PREDICTION  — returns (prediction, prob, shap_explanation)
# ======================================

# ======================================
# GLOBALLY TRUSTED DOMAINS
# URLs from these domains are capped at max 0.30 probability
# regardless of ML output — they are universally recognised
# platforms where even complex URL structures are normal.
# ======================================

TRUSTED_DOMAINS = {
    # Video / Social
    "youtube.com", "youtu.be", "vimeo.com",
    "facebook.com", "fb.com", "instagram.com",
    "twitter.com", "x.com", "linkedin.com",
    "tiktok.com", "reddit.com", "pinterest.com",
    # Productivity / Work
    "upwork.com", "fiverr.com", "freelancer.com",
    "slack.com", "trello.com", "notion.so",
    "zoom.us", "meet.google.com", "teams.microsoft.com",
    # Tech / Dev
    "github.com", "gitlab.com", "stackoverflow.com",
    "npmjs.com", "pypi.org", "docs.python.org",
    # Search / Cloud
    "google.com", "bing.com", "duckduckgo.com",
    "drive.google.com", "docs.google.com",
    "dropbox.com", "onedrive.live.com",
    # E-commerce / Finance
    "amazon.com", "ebay.com", "etsy.com",
    "paypal.com", "stripe.com", "shopify.com",
    # Education / Research
    "wikipedia.org", "scholar.google.com",
    "researchgate.net", "academia.edu",
    "coursera.org", "udemy.com", "edx.org",
    # News
    "bbc.com", "cnn.com", "reuters.com",
    "theguardian.com", "nytimes.com",
    # Microsoft / Apple / Major Tech
    "microsoft.com", "apple.com", "support.apple.com",
    "office.com", "live.com", "outlook.com",
    # Email
    "gmail.com", "mail.google.com", "proton.me",
}

def _get_registered_domain(url: str) -> str:
    """Return 'domain.tld' (registered domain) from a URL."""
    try:
        ext = tldextract.extract(url)
        return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        return ""


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

    prob = float(model.predict_proba(df)[0][1])

    # ── Layer 1: Trusted global domain cap ────────────────────────────
    # Globally recognised platforms are capped at 0.28 (never "Malicious")
    # even when their URLs look complex (e.g. YouTube query strings, Upwork paths).
    reg_domain = _get_registered_domain(url)
    if reg_domain in TRUSTED_DOMAINS:
        prob = min(prob, 0.28)

    else:
        # ── Layer 2: Smart multi-condition false-positive compensation ──
        # ALL 6 conditions must be true together — one failure disables the reduction.
        # This means exotic TLD / new domain / suspicious word alone blocks compensation.
        try:
            tld = tldextract.extract(url).suffix.lower()
        except Exception:
            tld = ""

        SAFE_TLDS = {"com", "org", "edu", "net", "gov", "io",
                     "co", "ac", "uk", "ca", "au", "de", "fr"}

        all_safe = (
            features.get("domain_age_days", 0)      > 730  and  # > 2 years old
            features.get("uses_https", 0)           == 1   and  # HTTPS
            features.get("suspicious_words", 0)     == 0   and  # no sus words
            features.get("dns_resolves", 0)         == 1   and  # DNS resolves
            features.get("checking_ip_address", 0)  == 0   and  # not raw IP
            tld in SAFE_TLDS                                     # standard TLD
        )
        if all_safe:
            prob = max(0.0, prob - 0.12)

        # ── Layer 3: Boost for genuinely malicious combinations ────────
        # Ensures truly dangerous URLs are not accidentally reduced.
        exotic_tld   = tld not in SAFE_TLDS and len(tld) >= 4
        new_domain   = features.get("domain_age_days", 0) < 30
        has_sus      = features.get("suspicious_words", 0) == 1
        many_dashes  = features.get("count_dash", 0) >= 3
        long_url     = features.get("url_length", 0) > 75
        random_path  = features.get("fd_length", 0) > 20

        malicious_signals = sum([exotic_tld, new_domain, has_sus,
                                 many_dashes, (long_url and has_sus), random_path])
        if malicious_signals >= 2:
            prob = min(1.0, prob + 0.08 * (malicious_signals - 1))

        # ── Legacy rule (kept) ─────────────────────────────────────────
        if new_domain and has_sus:
            prob = min(1.0, prob + 0.08)

    prediction       = 1 if prob >= 0.70 else 0
    shap_explanation = get_shap_explanation(df)

    return prediction, prob, shap_explanation

# ======================================
# CLI
# ======================================

if __name__ == "__main__":
    image_path = input("Enter QR image path: ")
    url        = extract_url_from_qr(image_path)
    if not url:
        print("No QR code detected.")
    else:
        print("\nExtracted URL:", url)
        if is_short_url(url):
            print("⚠️  SHORT URL DETECTED — destination is hidden.")
        prediction, probability, shap_exp = predict_url(url)
        print("Malicious Probability:", round(probability, 4))
        if probability < 0.30:
            print("✅ BENIGN")
        elif probability < 0.70:
            print("⚠️ SUSPICIOUS")
        else:
            print("🚨 MALICIOUS")
        if shap_exp:
            print("\nTop 5 Features (SHAP):")
            for item in shap_exp[:5]:
                print(f"  {item['feature']:30s} val={item['value']:8.2f}  shap={item['shap_value']:+.4f}")
