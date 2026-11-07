import os
import time
import requests
import threading
import re
import sys
import json
import math
from flask import Flask, render_template_string, jsonify, request
from urllib.parse import urlparse

app = Flask(__name__)

# ---- Config (override via env vars, or a `.env` file sourced by the launcher) ----
# Defaults are generic so the repo is safe to publish; put your real values in `.env`.
SPLASH_URL        = os.getenv("DASHSPLASH_SPLASH_URL", "http://127.0.0.1:8000/metrics")
MACMON_URL        = os.getenv("DASHSPLASH_MACMON_URL", "http://127.0.0.1:9090/json")
OLLAMA_URL        = os.getenv("DASHSPLASH_OLLAMA_URL", "http://127.0.0.1:11434/api/ps")
LLMSTUDIO_URL     = os.getenv("DASHSPLASH_LLMSTUDIO_URL", "http://192.168.1.x:1234")
OPENCODE_GO_USAGE_URL = os.getenv("DASHSPLASH_OPENCODE_GO_URL", "https://opencode.ai/zen/go/v1/usage")
MODEL_NAME        = os.getenv("DASHSPLASH_MODEL",      "Qwen3.8-27B-4bit")
GPU_TDP           = float(os.getenv("DASHSPLASH_GPU_TDP",  "100"))    # GPU TDP in W
MEMORY_TOTAL_GB   = float(os.getenv("DASHSPLASH_MEM_GB",  "54.5"))    # unified memory in GB
NODE_NAME         = os.getenv("DASHSPLASH_NODE_NAME",  "Mac Mini M5 Pro")
NODE_IP           = os.getenv("DASHSPLASH_NODE_IP",    "192.168.1.x")
PORT              = int(os.getenv("DASHSPLASH_PORT",   "8081"))
KWH_PRICE         = float(os.getenv("DASHSPLASH_KWH_PRICE", "0.13"))  # € per kWh

# Base URLs (host:port only) so we can hit control endpoints that live
# beside the metrics/ps paths. Used by the "release model" buttons.
def _base(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"
SPLASH_BASE = _base(SPLASH_URL)   # http://192.168.1.x:8000
OLLAMA_BASE = _base(OLLAMA_URL)   # http://192.168.1.x:11434

# Splash live status (small JSON: ready/thermal/reasoning_switch/
# scheduler counts/active_requests). Feeds the "thinking" indicator.
SPLASH_STATUS_URL = os.getenv("DASHSPLASH_STATUS_URL", f"{SPLASH_BASE}/status")

metrics_cache = {}
macmon_cache = {}
ollama_cache = {}
llmstudio_cache = {}
splash_status_cache = {}

# Estimate resident RAM (bytes) for a loaded model from its id + metadata.
# macmon/LM Studio expose no per-model RSS, so we estimate from the declared
# weights (params x bits) — this is the model's weight footprint, NOT the full
# process RSS (which adds KV-cache + runtime and is larger). Good enough to give
# LLM Studio its own segment in the memory bar.
_QUANT_BITS = {
    "4bit": 4, "8bit": 8, "16bit": 16, "bfloat16": 16, "float16": 16,
    "q4_k_m": 4, "q4_k_s": 4, "q4_0": 4, "q5_k_m": 5, "q8_0": 8,
    "af_4": 4, "af_16": 16, "mlx_4bit": 4, "mlx_8bit": 8, "none": 16,
}

def _model_params(id_str, arch=None, quant=None):
    """Best-effort total params (int) from a model id. Returns None if unparseable.
    Uses active params (A3B / MoE 'a' count) for MoE arches, else total."""
    s = (id_str or "").lower()
    arch = (arch or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)(?:[_.-]?(?:b|B)\b|b(?=[a-z0-9]))", s)
    total = None
    for mm in re.finditer(r"(\d+(?:\.\d+)?)[_.\- ]?(?:b|billion|B)\b", s):
        total = float(mm.group(1)) * 1e9
    if total is None:
        m2 = re.search(r"-(\d+(?:\.\d+)?)b", s)
        if m2:
            total = float(m2.group(1)) * 1e9
    # MoE active params: e.g. "35b-a3b" -> active 3B; "35b a3b"
    if total and ("moe" in arch or "a" in s):
        ma = re.search(r"a\s?(\d+(?:\.\d+)?)\s?b", s)
        if ma:
            return float(ma.group(1)) * 1e9
    return total

def _est_model_ram_bytes(mid, meta=None):
    """Estimated resident RAM (bytes) for one model. 0 if unknown.
    Uses TOTAL params (all MoE experts are resident in RAM even if only a
    subset is active), x declared quant bits."""
    meta = meta or {}
    s = (mid or "").lower()
    arch = (meta.get("arch") or "").lower()
    total = None
    # Take the LARGEST "N b" marker: for MoE the total (35b) exceeds the active
    # (a3b=3b), and only the total is resident in RAM. Max avoids "a3b" clobbering "35b".
    for mm in re.finditer(r"(\d+(?:\.\d+)?)[_.\- ]?(?:b|billion|B)\b", s):
        val = float(mm.group(1)) * 1e9
        total = val if total is None else max(total, val)
    if not total:
        total = _model_params(mid, arch, meta.get("quantization"))
    if not total:
        return 0
    q = (meta.get("quantization") or "").lower()
    bits = None
    for key, b in _QUANT_BITS.items():
        if key in q:
            bits = b
            break
    if bits is None:
        bits = 4  # most LM Studio models here are 4-bit
    return int(total * bits / 8)
_llmstudio_last_bench = 0
llmstudio_ids = {}
# OpenCode Go delegation-quota cache (percent used per window; no token counts).
_go_last_fetch = 0
_go_cache = {}

# ─── Energy history store (persisted power samples) ─────────────────────────
# Format: JSON Lines, one object per poll at data/energy/history.jsonl
#   {"t": <epoch seconds>, "sys": <W>, "gpu": <W>, "cpu": <W>}
# The update loop appends one row per poll (~5 s). Rows older than 24 h are
# thinned to one averaged point per minute and rows older than 90 days are
# dropped, so the file stays bounded (~150k lines worst case) and every range
# query is served from the in-memory mirror of that store.
# NOTE: this is a separate store from the in-memory `history` dict below,
# which only covers the running session and is NOT persisted.
_ENERGY_DIR = os.getenv(
    "DASHSPLASH_ENERGY_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "energy"))
_ENERGY_FILE = os.path.join(_ENERGY_DIR, "history.jsonl")
_ENERGY_LOCK = threading.Lock()
_ENERGY_SAMPLES = None          # [(t, sys_w, gpu_w, cpu_w)] ascending, lazy-loaded
_ENERGY_APPENDS = 0
_ENERGY_RETENTION_S = 90 * 86400   # keep 90 days
_ENERGY_FULL_RES_S = 24 * 3600     # full resolution for the last day
_ENERGY_THIN_S = 60                # bucket size (s) for older rows
_ENERGY_MAX_ROWS = 400000          # hard ceiling before a forced thinning
ENERGY_RANGES = {
    "1h": 3600,
    "24h": 86400,
    "7d": 7 * 86400,
    "30d": 30 * 86400,
    "90d": 90 * 86400,
}


def _energy_load():
    """Load history.jsonl into the in-memory list (idempotent, caller locks)."""
    global _ENERGY_SAMPLES
    if _ENERGY_SAMPLES is not None:
        return _ENERGY_SAMPLES
    rows = []
    try:
        os.makedirs(_ENERGY_DIR, exist_ok=True)
        if os.path.exists(_ENERGY_FILE):
            with open(_ENERGY_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        rows.append((float(d["t"]), float(d.get("sys", 0) or 0),
                                     float(d.get("gpu", 0) or 0), float(d.get("cpu", 0) or 0)))
                    except Exception:
                        continue        # skip a torn/corrupt line, keep the rest
    except Exception as e:
        print(f"[ENERGY] load failed ({_ENERGY_FILE}): {e}", flush=True)
    rows.sort(key=lambda r: r[0])
    _ENERGY_SAMPLES = rows
    return rows


def _energy_rewrite(rows):
    """Atomically replace history.jsonl with `rows` (caller holds the lock)."""
    try:
        os.makedirs(_ENERGY_DIR, exist_ok=True)
        tmp = _ENERGY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for t, sysw, gpuw, cpuw in rows:
                f.write(json.dumps({"t": round(t, 1), "sys": round(sysw, 2),
                                    "gpu": round(gpuw, 2), "cpu": round(cpuw, 2)}) + "\n")
        os.replace(tmp, _ENERGY_FILE)
    except Exception as e:
        print(f"[ENERGY] rewrite failed ({_ENERGY_FILE}): {e}", flush=True)


def _energy_thin(rows):
    """Drop >90d rows and merge the 24h..90d tail into 1-minute averages."""
    if not rows:
        return rows
    now = rows[-1][0]
    cut_full = now - _ENERGY_FULL_RES_S
    cut_ret = now - _ENERGY_RETENTION_S
    out = []
    bucket = None   # [bucket_start, sys_sum, gpu_sum, cpu_sum, n]

    def flush():
        nonlocal bucket
        if bucket:
            n = bucket[4]
            out.append((bucket[0], bucket[1] / n, bucket[2] / n, bucket[3] / n))
        bucket = None

    for t, sysw, gpuw, cpuw in rows:
        if t < cut_ret:
            continue
        if t >= cut_full:
            flush()
            out.append((t, sysw, gpuw, cpuw))
            continue
        k = int(t // _ENERGY_THIN_S) * _ENERGY_THIN_S
        if bucket is None or bucket[0] != k:
            flush()
            bucket = [k, sysw, gpuw, cpuw, 1]
        else:
            bucket[1] += sysw
            bucket[2] += gpuw
            bucket[3] += cpuw
            bucket[4] += 1
    flush()
    return out


def _energy_needs_thin(rows):
    """True when the store actually needs pruning/merging (cheap-ish check)."""
    if not rows:
        return False
    now = rows[-1][0]
    if rows[0][0] < now - _ENERGY_RETENTION_S:
        return True                      # something is older than 90 days
    cut_full = now - _ENERGY_FULL_RES_S
    old = sum(1 for r in rows if r[0] < cut_full)
    max_old = (_ENERGY_RETENTION_S - _ENERGY_FULL_RES_S) // _ENERGY_THIN_S + 2
    return old > max_old * 1.2


def energy_record(now, macmon):
    """Append one power sample. Called once per poll from record_history().

    Skips the poll when macmon returned nothing (macmon down): writing zeros
    would silently invent consumption history.
    """
    global _ENERGY_APPENDS
    if not isinstance(macmon, dict) or "sys_power" not in macmon:
        return
    try:
        row = (float(now),
               float(macmon.get("sys_power") or 0),
               float(macmon.get("gpu_power") or 0),
               float(macmon.get("cpu_power") or 0))
    except Exception:
        return
    if row[0] <= 0:
        return
    with _ENERGY_LOCK:
        rows = _energy_load()
        # never store two samples inside the same second (restart overlap)
        if rows and abs(row[0] - rows[-1][0]) < 1.0:
            return
        rows.append(row)
        _ENERGY_APPENDS += 1
        if (_ENERGY_APPENDS % 360 == 0 or len(rows) > _ENERGY_MAX_ROWS) and _energy_needs_thin(rows):
            thinned = _energy_thin(rows)
            rows[:] = thinned
            _energy_rewrite(rows)
            return          # the rewrite already persisted this sample
        try:
            os.makedirs(_ENERGY_DIR, exist_ok=True)
            with open(_ENERGY_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": round(row[0], 1), "sys": round(row[1], 2),
                                    "gpu": round(row[2], 2), "cpu": round(row[3], 2)}) + "\n")
        except Exception as e:
            print(f"[ENERGY] append failed ({_ENERGY_FILE}): {e}", flush=True)


def _energy_query(range_key, max_points=500):
    """Downsampled points + integrated kWh/cost for one range.

    Returns {"range", "start", "end", "samples", "points", "kwh", "cost"} where
    points is [{t: epoch_s, watts, gpu_watts}] with at most ~max_points entries.
    """
    span = ENERGY_RANGES.get(range_key)
    if not span:
        return None
    with _ENERGY_LOCK:
        rows = list(_energy_load())
    if not rows:
        return {"range": range_key, "start": None, "end": None, "samples": 0,
                "points": [], "kwh": 0.0, "cost": 0.0}
    end = rows[-1][0]
    start = end - span
    win = [r for r in rows if r[0] >= start]
    if not win:
        return {"range": range_key, "start": start, "end": end, "samples": 0,
                "points": [], "kwh": 0.0, "cost": 0.0}
    # Trapezoid integration on the raw rows of the window (system watts).
    kwh = 0.0
    for i in range(1, len(win)):
        dt_h = (win[i][0] - win[i - 1][0]) / 3600.0
        if dt_h <= 0:
            continue
        kwh += ((win[i][1] + win[i - 1][1]) / 2.0) * dt_h / 1000.0
    n = len(win)
    step = max(1, int(math.ceil(n / float(max_points))))
    pts = []
    for i in range(0, n, step):
        chunk = win[i:i + step]
        m = len(chunk)
        mid = chunk[m // 2]
        pts.append({
            "t": round(mid[0], 1),
            "watts": round(sum(c[1] for c in chunk) / m, 1),
            "gpu_watts": round(sum(c[2] for c in chunk) / m, 1),
        })
    return {"range": range_key, "start": start, "end": end, "samples": n,
            "points": pts, "kwh": round(kwh, 4), "cost": round(kwh * KWH_PRICE, 4)}


history = {
    'memory_used': [], 'memory_limit': [],
    'decode_tps': [], 'prefill_tps': [],
    'ttft_p50': [], 'itl_p50': [],
    'gpu_temp': [], 'cpu_temp': [],
    'gpu_power': [], 'sys_power': [],
    'gpu_util': [], 'gpu_freq': [],
    'fan_rpm': [],
}

def fetch_llmstudio():
    """LLM Studio is OpenAI-compatible but exposes no live T/s. We probe it
    (throttled, bench every 120s) and accumulate completion tokens per model.

    We read the model list from /api/v0/models (LM Studio native) because it
    carries `state` ("loaded"/"not-loaded"), so we only surface models that
    are actually resident in memory. If that endpoint is absent, we fall back
    to /v1/models and treat every listed model as loaded.
    """
    global _llmstudio_last_bench
    now = time.time()
    base = LLMSTUDIO_URL.rstrip('/')
    ids = []
    loaded = []          # list of (mid, meta_dict) for state=='loaded'
    try:
        r = requests.get(f"{base}/api/v0/models", timeout=6)
        for m in r.json().get("data", []):
            if not isinstance(m, dict):
                continue
            mid = m.get("id")
            if not mid:
                continue
            ids.append(mid)
            if m.get("state") == "loaded":
                loaded.append((mid, {"arch": m.get("arch"), "quantization": m.get("quantization")}))
    except Exception:
        ids = []
    if not ids:
        # fallback: OpenAI /v1/models (no state/meta) -> treat all as loaded
        try:
            r = requests.get(f"{base}/v1/models", timeout=6)
            for m in r.json().get("data", []):
                if not isinstance(m, dict) or not m.get("id"):
                    continue
                ids.append(m.get("id"))
                loaded.append((m.get("id"), {"arch": m.get("arch"), "quantization": m.get("quantization")}))
        except Exception:
            return {"online": False, "count": 0, "models": [], "tps": 0, "ram_used": 0}
        loaded = [(mid, {}) for mid in ids]
    loaded_ids = [mid for mid, _ in loaded]
    if now - _llmstudio_last_bench >= 120 and loaded_ids:
        _llmstudio_last_bench = now
        for mid in loaded_ids:
            if mid not in llmstudio_ids:
                llmstudio_ids[mid] = {"gen": 0, "tps": 0}
            try:
                t0 = time.time()
                rr = requests.post(
                    f"{base}/v1/chat/completions",
                    json={"model": mid, "messages": [{"role": "user", "content": "di hola"}],
                          "max_tokens": 8, "stream": False}, timeout=60)
                u = (rr.json().get("usage") or {}).get("completion_tokens", 0)
                llmstudio_ids[mid]["gen"] = llmstudio_ids[mid].get("gen", 0) + u
                wall = max(time.time() - t0, 0.01)
                if u:
                    llmstudio_ids[mid]["tps"] = round(u / wall, 1)
            except Exception:
                pass
    models = []
    meta_by_id = {mid: meta for mid, meta in loaded}
    ram_total = 0
    for m in loaded_ids:
        st = llmstudio_ids.get(m, {})
        ram = _est_model_ram_bytes(m, meta_by_id.get(m))
        ram_total += ram
        models.append({"id": m, "gen": st.get("gen", 0), "tps": st.get("tps", 0), "ram": ram})
    return {"online": True, "count": len(models),
            "models": models,
            "ram_used": ram_total,
            "tps": round(sum(v.get("tps", 0) for v in llmstudio_ids.values()), 1)}


def fetch_opencodego():
    """OpenCode Go delegation quota.

    The API exposes quota windows only (no token counters), each with
    `status`, `percent` (fraction of the window's quota used, 0-100) and
    `resetsAt` (ISO8601). Throttled at 120s and cached so the update loop can
    call it every 5s without hammering the endpoint. Never raises: any failure
    yields {'online': False, ...} and the dashboard renders a muted line.
    """
    global _go_last_fetch, _go_cache
    now = time.time()
    if _go_cache and (now - _go_last_fetch) < 120:
        return _go_cache
    key = (os.getenv('DASHSPLASH_OPENCODE_GO_KEY') or '').strip()
    if not key:
        _go_cache = {'online': False, 'reason': 'no_api_key', 'windows': {}}
        _go_last_fetch = now
        return _go_cache
    try:
        r = requests.get(
            OPENCODE_GO_USAGE_URL,
            headers={'Authorization': f'Bearer {key}'},
            timeout=8)
        if r.status_code != 200:
            _go_cache = {'online': False, 'reason': f'http_{r.status_code}', 'windows': {}}
            _go_last_fetch = now
            return _go_cache
        payload = r.json()
        usage = payload.get('usage') if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            _go_cache = {'online': False, 'reason': 'bad_payload', 'windows': {}}
            _go_last_fetch = now
            return _go_cache
        windows = {}
        for name in ('rolling', 'weekly', 'monthly'):
            w = usage.get(name)
            if not isinstance(w, dict):
                w = {}
            raw = w.get('percent')
            pct = None
            if raw is not None:
                try:
                    pct = float(raw)
                except (TypeError, ValueError):
                    pct = None
            windows[name] = {
                'status': w.get('status'),
                'percent': pct,
                'resetsAt': w.get('resetsAt'),
            }
        _go_cache = {'online': True, 'windows': windows,
                     'fetched_at': int(now), 'reason': None}
        _go_last_fetch = now
        return _go_cache
    except Exception as e:
        _go_cache = {'online': False, 'reason': str(e)[:120], 'windows': {}}
        _go_last_fetch = now
        return _go_cache


def parse_splash(text):
    data = {}
    labels = {}
    for line in text.split('\n'):
        if not line or line.startswith('#'):
            continue
        m = re.match(r'^(\S+?)(?:\{(.+?)\})?\s+([-\d.eE+]+)', line)
        if m:
            key = m.group(1)
            value = float(m.group(3))
            if m.group(2):
                labels[key] = {}
                for lm in re.finditer(r'(\w+)="([^"]*)"', m.group(2)):
                    labels[key][lm.group(1)] = lm.group(2)
                    data[f"{key}.{lm.group(1)}"] = value
            else:
                data[key] = value
    return data, labels

def extract_splash(data, labels):
    m = {}
    m['mem_current'] = data.get('splash_memory_current_bytes', 0)
    m['mem_limit'] = data.get('splash_memory_limit_bytes', MEMORY_TOTAL_GB * 1e9)
    m['mem_headroom'] = data.get('splash_memory_headroom_bytes', 0)
    m['mem_peak'] = data.get('splash_memory_peak_bytes', 0)
    for state in ['nominal', 'fair', 'serious', 'critical']:
        key = f'splash_thermal_state.{state}'
        if data.get(key, 0) == 1:
            m['thermal_state'] = state
            break
    else:
        m['thermal_state'] = 'nominal'
    m['decode_tps'] = data.get('splash_decode_tokens_per_second', 0)
    m['prefill_tps'] = data.get('splash_prefill_tokens_per_second', 0)
    m['ttft_p50'] = data.get('splash_ttft_p50_milliseconds', 0)
    m['ttft_p95'] = data.get('splash_ttft_p95_milliseconds', 0)
    m['itl_p50'] = data.get('splash_itl_p50_milliseconds', 0)
    m['itl_p95'] = data.get('splash_itl_p95_milliseconds', 0)
    m['scheduler_decoding'] = int(data.get('splash_scheduler_decoding', 0))
    m['scheduler_prefilling'] = int(data.get('splash_scheduler_prefilling', 0))
    m['scheduler_queued'] = int(data.get('splash_scheduler_queued', 0))
    m['kv_pages_allocated'] = int(data.get('splash_kv_pages_allocated', 0))
    m['kv_pages_active'] = int(data.get('splash_kv_pages_active', 0))
    m['kv_pages_cache'] = int(data.get('splash_kv_pages_cache', 0))
    m['kv_allocated_bytes'] = data.get('splash_kv_allocated_bytes', 0)
    m['requests_submitted'] = int(data.get('splash_requests_submitted_total', 0))
    m['requests_completed'] = int(data.get('splash_requests_completed_total', 0))
    m['requests_failed'] = int(data.get('splash_requests_failed_total', 0))
    m['requests_cancelled'] = int(data.get('splash_requests_cancelled_total', 0))
    m['prefill_tokens'] = int(data.get('splash_prefill_input_tokens_total', 0))
    m['decode_tokens'] = int(data.get('splash_decode_output_tokens_total', 0))
    m['drafted_tokens'] = int(data.get('splash_drafted_tokens_total', 0))
    m['accepted_draft'] = int(data.get('splash_accepted_draft_tokens_total', 0))
    m['cache_hits'] = int(data.get('splash_cache_hits_total', 0))
    m['cache_cold_misses'] = int(data.get('splash_cache_cold_misses_total', 0))
    m['cache_reused_tokens'] = int(data.get('splash_cache_reused_tokens_total', 0))
    m['thermal_state'] = m.get('thermal_state', 'nominal')
    m['runtime'] = labels.get('splash_info', {}).get('runtime', 'native')
    return m

def extract_macmon(data):
    m = {}
    temp = data.get('temp', {})
    m['gpu_temp'] = temp.get('gpu_temp_avg', 0)
    m['cpu_temp'] = temp.get('cpu_temp_avg', 0)
    m['gpu_power'] = data.get('gpu_power', 0)
    m['cpu_power'] = data.get('cpu_power', 0)
    m['sys_power'] = data.get('sys_power', 0)
    m['all_power'] = data.get('all_power', 0)
    gpu_usage = data.get('gpu_usage')
    m['gpu_util'] = (gpu_usage[-1] if isinstance(gpu_usage, list) and len(gpu_usage) > 0 else data.get('gpu_scaled_ratio', 0))
    m['gpu_freq'] = data.get('gpu_freq_mhz', 0)
    m['cpu_util'] = data.get('cpu_usage_pct', 0)
    m['cpu_freq'] = data.get('ecpu_freq_mhz', 0)
    fans = data.get('fans', [])
    m['fan_rpm'] = fans[0].get('rpm', 0) if isinstance(fans, list) and len(fans) > 0 else 0
    # RAM real del sistema (fuente de verdad, no el techo de splash)
    mem = data.get('memory', {})
    m['mem_total'] = mem.get('ram_total', int(MEMORY_TOTAL_GB * 1e9))
    m['mem_used']  = mem.get('ram_usage', 0)
    return m

def record_history(splash, macmon):
    now = time.time()
    history['memory_used'].append((now, splash.get('mem_current', 0) / 1e9))
    history['memory_limit'].append((now, splash.get('mem_limit', 1) / 1e9))
    history['decode_tps'].append((now, splash.get('decode_tps', 0)))
    history['prefill_tps'].append((now, splash.get('prefill_tps', 0)))
    history['ttft_p50'].append((now, splash.get('ttft_p50', 0)))
    history['itl_p50'].append((now, splash.get('itl_p50', 0)))
    history['gpu_temp'].append((now, macmon.get('gpu_temp', 0)))
    history['cpu_temp'].append((now, macmon.get('cpu_temp', 0)))
    history['gpu_power'].append((now, macmon.get('gpu_power', 0)))
    history['sys_power'].append((now, macmon.get('sys_power', 0)))
    history['gpu_util'].append((now, macmon.get('gpu_util', 0)))
    history['gpu_freq'].append((now, macmon.get('gpu_freq', 0)))
    history['fan_rpm'].append((now, macmon.get('fan_rpm', 0)))
    # Persist the power samples for the Energy card's multi-range history.
    try:
        energy_record(now, macmon)
    except Exception as _e:
        print(f"[ENERGY] record failed: {_e}", flush=True)
    cutoff = now - 3600
    for k in history:
        history[k] = [(t, v) for t, v in history[k] if t > cutoff]

def fetch_data():
    try:
        r = requests.get(SPLASH_URL, timeout=3)
        d, l = parse_splash(r.text)
        return extract_splash(d, l)
    except:
        return {}

def fetch_macmon():
    try:
        r = requests.get(MACMON_URL, timeout=3)
        return extract_macmon(r.json())
    except:
        return {}

def fetch_splash_status():
    """Live /status snapshot: ready, thermal, reasoning switch, scheduler
    counts and the per-request list (id/phase/token counters). Falls back to
    the last good cache on failure so the UI never flickers to empty."""
    try:
        r = requests.get(SPLASH_STATUS_URL, timeout=3)
        d = r.json()
        sched = d.get('scheduler') or {}
        reqs = []
        for a in (d.get('active_requests') or []):
            if not isinstance(a, dict):
                continue
            reqs.append({
                'id': a.get('id'),
                'phase': a.get('phase') or 'decode',
                'prompt_tokens': a.get('prompt_tokens', 0) or 0,
                'prompt_processed': a.get('prompt_processed', 0) or 0,
                'generated_tokens': a.get('generated_tokens', 0) or 0,
                'max_new_tokens': a.get('max_new_tokens', 0) or 0,
                'age_ms': int(a.get('age_ms', 0) or 0),
            })
        return {
            'ok': True,
            'ready': bool(d.get('ready', False)),
            'thermal_state': d.get('thermal_state') or 'nominal',
            'reasoning_switch': bool(d.get('reasoning_switch', False)),
            'scheduler': {
                'decoding': sched.get('decoding', 0) or 0,
                'prefilling': sched.get('prefilling', 0) or 0,
                'queued': sched.get('queued', 0) or 0,
            },
            'active_requests': reqs,
            'updated': time.time(),
        }
    except Exception:
        return dict(splash_status_cache) or {'ok': False}

def extract_ollama(data):
    models = data.get('models', [])
    total_vram = sum(m.get('size_vram', 0) for m in models)
    total_ram  = sum(m.get('size', 0) for m in models)
    m = {}
    m['online']   = bool(models)
    m['count']    = len(models)
    m['vram_used'] = total_vram
    m['ram_used']  = total_ram
    m['models'] = [
        {
            'name':    (m2.get('name') or m2.get('model') or '?'),
            'vram':    m2.get('size_vram', 0),
            'ram':     m2.get('size', 0),
            'details': m2.get('details', {}),
            'expires': m2.get('expires_at', ''),
        } for m2 in models
    ]
    return m

def fetch_ollama():
    try:
        r = requests.get(OLLAMA_URL, timeout=3)
        return extract_ollama(r.json())
    except:
        return {'online': False, 'count': 0, 'vram_used': 0, 'ram_used': 0, 'models': []}

_last_splash_snapshot = {}

def compute_splash_delta(s):
    """Per-poll deltas of aggregate counters → approx per-request stats."""
    global _last_splash_snapshot
    last = _last_splash_snapshot
    if not last:
        _last_splash_snapshot = {k: s.get(k, 0) for k in
            ('requests_submitted', 'requests_completed', 'prefill_tokens', 'decode_tokens', 'cache_reused_tokens')}
        s['splash_delta'] = None
        return s
    d = {
        'submitted': max(0, s.get('requests_submitted', 0) - last.get('requests_submitted', 0)),
        'completed': max(0, s.get('requests_completed', 0) - last.get('requests_completed', 0)),
        'input': max(0, s.get('prefill_tokens', 0) - last.get('prefill_tokens', 0)),
        'output': max(0, s.get('decode_tokens', 0) - last.get('decode_tokens', 0)),
        'cached': max(0, s.get('cache_reused_tokens', 0) - last.get('cache_reused_tokens', 0)),
    }
    _last_splash_snapshot = {k: s.get(k, 0) for k in
        ('requests_submitted', 'requests_completed', 'prefill_tokens', 'decode_tokens', 'cache_reused_tokens')}
    s['splash_delta'] = d
    return s

def update_loop():
    global metrics_cache, macmon_cache, ollama_cache, llmstudio_cache, splash_status_cache
    while True:
        try:
            splash = fetch_data()
            splash = compute_splash_delta(splash)
            macmon = fetch_macmon()
            ollama = fetch_ollama()
            record_history(splash, macmon)
            metrics_cache['splash'] = splash
            macmon_cache['macmon'] = macmon
            ollama_cache['ollama'] = ollama
        except:
            pass
        # Live status (/status) — small payload, same 5s cadence as the rest.
        try:
            st = fetch_splash_status()
            if st.get('ok'):
                splash_status_cache['splash_status'] = st
        except:
            pass
        # LLM Studio is on a reachable endpoint independent of splash/ollama, so
        # always update its cache even if the block above throws.
        try:
            llmstudio = fetch_llmstudio()
            if llmstudio:
                llmstudio_cache['llmstudio'] = llmstudio
        except:
            pass
        # OpenCode Go quota is self-throttled (120s) and never raises.
        try:
            fetch_opencodego()
        except:
            pass
        time.sleep(5)

threading.Thread(target=update_loop, daemon=True).start()

DASHBOARD = '''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mac Monitor</title>
<style>
/* ===== THEME TOKENS =====
   :root stays the DARK theme; the same tokens are re-applied under
   html[data-theme="dark"] so the attribute always wins, and
   html[data-theme="light"] overrides them with a light palette.
   Every colour in this stylesheet comes from these tokens. */
:root, html[data-theme="dark"] {
    --bg: #0f1117;
    --card: #181c25;
    --card-alt: #1e2330;
    --border: #2a2f3a;
    --text: #e8eaed;
    --text-dim: #9aa0a6;
    --text-mut: #5f6368;
    --green: #34a853;
    --green-bg: rgba(52,168,83,0.12);
    --orange: #f9ab00;
    --orange-bg: rgba(249,171,0,0.12);
    --purple: #a144fb;
    --purple-bg: rgba(161,68,251,0.12);
    --blue: #4285f4;
    --blue-bg: rgba(66,133,244,0.12);
    --red: #ea4335;
    --red-bg: rgba(234,67,53,0.12);
    --cream: #2d2a1f;
    --cream-border: #3d3825;
    /* structure / component tokens (theme-aware) */
    --track: #2a2f3a;
    --seg-free: #3a3f4a;
    --seg-free-legend: #5a6070;
    --teal: #14b8a6;
    --model-name: #e8d5a3;
    --hover: rgba(255,255,255,0.06);
    --hover-mid: rgba(255,255,255,0.10);
    --hover-strong: rgba(255,255,255,0.16);
    --overlay: rgba(0,0,0,0.6);
    --danger-text: #ff8a8a;
    --danger-bg: rgba(255,120,120,0.12);
    --danger-bg-hover: rgba(255,120,120,0.28);
    --danger-border: rgba(255,120,120,0.35);
    --ok-soft: #4ade80;
    --crit-soft: #ef4444;
    --green-border: rgba(52,168,83,0.45);
    --panel-shadow: 0 16px 40px rgba(0,0,0,0.50);
    --panel-bg: #181c25;
    --chart-grid: #2a2f3a;
    --chart-axis: #5f6368;
    --chart-fill: rgba(255,255,255,0.03);
    color-scheme: dark;
}
html[data-theme="light"] {
    --bg: #eef1f5;
    --card: #ffffff;
    --card-alt: #f5f7fa;
    --border: #d7dde5;
    --text: #16202b;
    --text-dim: #53616f;
    --text-mut: #8592a0;
    --green: #1e8e3e;              /* same OK accent, tuned for white bg */
    --green-bg: rgba(30,142,62,0.12);
    --orange: #c77c00;
    --orange-bg: rgba(199,124,0,0.12);
    --purple: #7c3aed;
    --purple-bg: rgba(124,58,237,0.10);
    --blue: #2563eb;
    --blue-bg: rgba(37,99,235,0.10);
    --red: #d93025;
    --red-bg: rgba(217,48,37,0.10);
    --cream: #fff8e6;
    --cream-border: #efe0b6;
    --track: #e4e9ef;
    --seg-free: #cdd5df;
    --seg-free-legend: #9aa5b1;
    --teal: #0f9488;
    --model-name: #8a6414;
    --hover: rgba(15,25,35,0.05);
    --hover-mid: rgba(15,25,35,0.08);
    --hover-strong: rgba(15,25,35,0.12);
    --overlay: rgba(15,25,35,0.35);
    --danger-text: #b3261e;
    --danger-bg: rgba(217,48,37,0.08);
    --danger-bg-hover: rgba(217,48,37,0.18);
    --danger-border: rgba(217,48,37,0.35);
    --ok-soft: #15803d;
    --crit-soft: #dc2626;
    --green-border: rgba(30,142,62,0.45);
    --panel-shadow: 0 16px 40px rgba(15,25,35,0.18);
    --panel-bg: #ffffff;
    --chart-grid: #dde3ea;         /* charts stay readable on white */
    --chart-axis: #6b7280;
    --chart-fill: rgba(15,25,35,0.03);
    color-scheme: light;
}
* { margin:0; padding:0; box-sizing:border-box; }

/* === HEADER TOOLBAR + ADMIN PANEL (theme-aware) === */
.header-right { display: flex; flex-direction: column; align-items: flex-end; gap: 8px; }
.header-tools { display: flex; align-items: center; gap: 6px; }
.icon-btn {
    display: inline-flex; align-items: center; justify-content: center;
    width: 30px; height: 28px; font-size: 14px; line-height: 1;
    color: var(--text-dim); background: var(--hover-mid);
    border: 1px solid var(--border); border-radius: 7px;
    cursor: pointer; text-decoration: none; transition: background .15s, color .15s, border-color .15s;
}
.icon-btn:hover { background: var(--hover-strong); color: var(--text); }
.icon-btn.active { color: var(--green); border-color: var(--green-border); background: var(--green-bg); }

.admin-hidden { display: none !important; }
.admin-panel {
    position: fixed; top: 58px; right: 20px; z-index: 900;
    width: 380px; max-width: calc(100vw - 32px); max-height: calc(100vh - 82px);
    overflow-y: auto;
    background: var(--panel-bg); color: var(--text);
    border: 1px solid var(--border); border-radius: 12px;
    box-shadow: var(--panel-shadow);
    animation: adminSlide .16s ease-out;
}
@keyframes adminSlide { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: none; } }
.admin-panel-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; border-bottom: 1px solid var(--border);
    position: sticky; top: 0; background: var(--panel-bg); z-index: 1;
}
.admin-panel-title { font-size: 14px; font-weight: 700; letter-spacing: .2px; }
.admin-close {
    width: 26px; height: 26px; border-radius: 6px; cursor: pointer;
    font-size: 13px; line-height: 1; color: var(--text-dim);
    background: var(--hover-mid); border: 1px solid var(--border);
}
.admin-close:hover { background: var(--hover-strong); color: var(--text); }
.admin-panel-body { padding: 8px 16px 16px; }
.admin-sec { padding: 12px 0; border-bottom: 1px solid var(--border); }
.admin-sec:last-child { border-bottom: none; }
.admin-sec-title {
    font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
    color: var(--text-dim); margin-bottom: 8px;
}
.admin-slot { font-size: 12px; color: var(--text-dim); }
.admin-slot:empty::before { content: 'Sin opciones'; font-size: 11px; color: var(--text-mut); font-style: italic; }
.admin-slot label { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; color: var(--text-dim); }
.admin-slot input[type="range"] { flex: 1; accent-color: var(--green); }
.admin-slot select, .admin-slot input[type="text"], .admin-slot input[type="number"] {
    background: var(--card-alt); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 8px; font-size: 12px;
}
.admin-slot button {
    background: var(--hover-mid); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px; font-size: 11px; cursor: pointer;
}
.admin-slot button:hover { background: var(--hover-strong); }
/* === CARD LAYOUT (visibilidad / columnas) — aplicado por applyCardLayout() === */
.stats-row[data-mode="onecol"], .bottom-row[data-mode="onecol"], .charts-row[data-mode="onecol"] {
    grid-template-columns: minmax(0, 1fr) !important;
}
.stats-row[data-mode="twocol"], .bottom-row[data-mode="twocol"], .charts-row[data-mode="twocol"] {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
}
.admin-slot .ad-note { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); margin: 10px 0 4px; }
.admin-slot .ad-note:first-child { margin-top: 0; }
.admin-slot .ad-modes { display: flex; flex-wrap: wrap; gap: 2px 16px; }
.admin-slot .ad-check, .admin-slot .ad-radio { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; color: var(--text-dim); cursor: pointer; }
.admin-slot .ad-check input, .admin-slot .ad-radio input { accent-color: var(--blue); margin: 0; cursor: pointer; }
.admin-slot .ad-order-row { display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 12px; color: var(--text-dim); }
.admin-slot .ad-order-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.admin-slot .ad-order-btns { display: flex; gap: 4px; }
.admin-slot .ad-mini { padding: 1px 7px; font-size: 11px; line-height: 1.4; min-width: 24px; }
.admin-slot .ad-mini:disabled { opacity: .35; cursor: default; }
.admin-slot .ad-actions { margin-top: 10px; }

@media (max-width: 520px) {
    .admin-panel { right: 8px; left: 8px; width: auto; }
}
body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 20px 24px; min-height: 100vh;
}

/* === CUSTOM BACKGROUND IMAGE (admin → Fondo) ===
   --bg-image is set by JS (setBg) to url('/bg/<file>') or removed.
   body.has-bg only turns the photo on; cards get a glassy, still-legible
   surface because the photo would otherwise bleed through them. */
body {
    background-image: var(--bg-image, none);
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
}
body.has-bg { --card-glass: rgba(24,28,37,0.72); --card-glass-alt: rgba(30,35,48,0.72); }
html[data-theme="light"] body.has-bg { --card-glass: rgba(255,255,255,0.80); --card-glass-alt: rgba(245,247,250,0.80); }
body.has-bg .node-card,
body.has-bg .stat-card,
body.has-bg .bottom-card,
body.has-bg .chart-card,
body.has-bg .modal {
    background: var(--card-glass) !important;
    -webkit-backdrop-filter: blur(12px) saturate(1.15);
    backdrop-filter: blur(12px) saturate(1.15);
}
body.has-bg .header { text-shadow: 0 1px 3px rgba(0,0,0,0.45); }
html[data-theme="light"] body.has-bg .header { text-shadow: 0 1px 3px rgba(255,255,255,0.6); }
/* admin panel stays opaque-ish so controls remain readable over a photo */
body.has-bg .admin-panel { background: var(--card-glass-alt, var(--panel-bg)); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); }

/* bg section controls */
.bg-thumb {
    display: block; width: 100%; height: 96px; object-fit: cover;
    border-radius: 8px; border: 1px solid var(--border); margin: 6px 0 8px;
    background: var(--card-alt);
}
.bg-thumb.bg-empty { display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: var(--text-mut); font-style: italic; }
.bg-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.bg-file-label {
    display: inline-block; cursor: pointer; user-select: none;
    background: var(--hover-mid); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 12px; font-size: 11px; font-weight: 600;
}
.bg-file-label:hover { background: var(--hover-strong); }
.bg-file-label input[type="file"] { display: none; }
.bg-status { font-size: 11px; color: var(--text-mut); min-height: 14px; }

/* === HEADER === */
.header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 16px;
}
.header-left h1 { font-size: 22px; font-weight: 700; color: var(--text); }
.header-sub {
    display: flex; align-items: center; gap: 8px; margin-top: 4px;
    font-size: 13px; color: var(--text-dim);
}
.header-sub .badge-online {
    background: var(--green-bg); color: var(--green);
    padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600;
}
.header-right {
    font-size: 12px; color: var(--text-dim); text-align: right;
}

/* === NODE CARD === */
.node-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 16px;
}
.node-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 14px; padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.node-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); box-shadow: 0 0 6px var(--green); }
.node-name { font-size: 15px; font-weight: 700; }
.node-badge {
    font-size: 10px; font-weight: 700; padding: 2px 6px;
    border-radius: 4px; text-transform: uppercase; letter-spacing: 0.5px;
}
.node-badge.head { background: var(--orange-bg); color: var(--orange); }
.node-badge.worker { background: var(--hover); color: var(--text-dim); }
.node-ip { margin-left: auto; font-size: 11px; color: var(--text-mut); }

/* Memory section */
.mem-section { margin-bottom: 12px; }
.mem-label { font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
.mem-free { font-size: 14px; font-weight: 700; margin-bottom: 6px; }
.mem-bar { height: 8px; background: var(--track); border-radius: 4px; overflow: hidden; display: flex; }
.mem-bar-splash { background: var(--green); height: 100%; }
.mem-bar-ollama { background: var(--blue); height: 100%; }
.mem-bar-llmstudio { background: var(--teal); height: 100%; }
.mem-bar-sys { background: var(--purple); height: 100%; }
.mem-bar-free { background: var(--seg-free); height: 100%; }
.mem-legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 5px; font-size: 10px; color: var(--text-dim); }
.mem-legend span::before { content: '● '; }
.mem-legend .lg-splash::before { color: var(--green); }
.mem-legend .lg-ollama::before { color: var(--blue); }
.mem-legend .lg-llmstudio::before { color: var(--teal); }
.mem-legend .lg-sys::before { color: var(--purple); }
.mem-legend .lg-free::before { color: var(--seg-free-legend); }

/* GPU metrics row */
.gpu-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }
.gpu-item { }
.gpu-label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.gpu-val { font-size: 16px; font-weight: 700; margin-bottom: 5px; }
.gpu-val .unit { font-size: 11px; font-weight: 400; color: var(--text-dim); }
.bar { height: 6px; background: var(--track); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
.bar-fill.orange { background: var(--orange); }
.bar-fill.green { background: var(--green); }
.bar-fill.purple { background: var(--purple); }
.bar-fill.blue { background: var(--blue); }

/* Decode/Prefill big metrics */
.perf-row {
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    margin-bottom: 10px; padding: 12px 16px;
    background: var(--card-alt); border-radius: 8px;
}
.perf-item .perf-label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.perf-item .perf-val { font-size: 22px; font-weight: 700; }
.perf-item .perf-val.orange { color: var(--orange); }
.perf-item .perf-val.purple { color: var(--purple); }
.perf-item .perf-val .unit { font-size: 12px; font-weight: 400; color: var(--text-dim); }
.perf-badge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 10px; color: var(--green); background: var(--green-bg);
    padding: 2px 8px; border-radius: 10px; margin-top: 6px;
}
.perf-badge::before { content: '●'; font-size: 7px; }

/* Divider */
.divider { height: 1px; background: var(--border); margin: 14px 0; }

/* Model box */
.model-box {
    display: flex; align-items: center; gap: 10px;
    background: var(--cream); border: 1px solid var(--cream-border);
    border-radius: 8px; padding: 10px 14px;
}
.model-name { font-size: 13px; font-weight: 600; color: var(--model-name); }
.model-ctx { font-size: 10px; color: var(--text-dim); background: var(--hover); padding: 2px 6px; border-radius: 4px; }
.release-btn {
    margin-left: 8px; cursor: pointer; user-select: none;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    color: var(--danger-text); background: var(--danger-bg);
    border: 1px solid var(--danger-border); border-radius: 6px;
    padding: 4px 9px; transition: all 0.15s;
}
.release-btn:hover { background: var(--danger-bg-hover); }
.release-btn:disabled { opacity: 0.5; cursor: wait; }

/* === AGGREGATE STATS === */
.stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }
.stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
}
.stat-label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.stat-value { font-size: 28px; font-weight: 700; line-height: 1.1; }
.stat-value .unit { font-size: 13px; font-weight: 400; color: var(--text-dim); }
.stat-sub { font-size: 10px; color: var(--text-mut); margin-top: 4px; }
.stat-spark { margin-top: 8px; height: 28px; }
.stat-spark canvas { width: 100%; height: 28px; }
.stat-bar { height: 5px; background: var(--track); border-radius: 3px; margin-top: 8px; overflow: hidden; }
.stat-bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

/* === BOTTOM ROW === */
.bottom-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
.bottom-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
}
.bc-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
}
.bc-title { font-size: 13px; font-weight: 600; }
.bc-badge { font-size: 10px; color: var(--text-dim); background: var(--hover); padding: 2px 6px; border-radius: 4px; }
.bc-btn { font-size: 10px; color: var(--text-dim); background: var(--hover-mid); border: none; padding: 3px 8px; border-radius: 4px; cursor: pointer; margin-left: auto; }
.bc-btn:hover { background: var(--hover-strong); }
.bc-big { font-size: 26px; font-weight: 700; }
.bc-big .unit { font-size: 12px; font-weight: 400; color: var(--text-dim); }
/* Energy card: range switcher (1h / 24h / 1sem / 1mes / 3meses) */
.energy-ranges { display: inline-flex; gap: 3px; margin-left: auto; }
.energy-range-btn {
    font-size: 9px; line-height: 1.3; padding: 2px 5px; cursor: pointer;
    border-radius: 4px; border: 1px solid var(--border);
    background: var(--hover-mid); color: var(--text-dim); font-family: inherit;
}
.energy-range-btn:hover { background: var(--hover-strong); color: var(--text); }
.energy-range-btn.active { background: var(--orange-bg); color: var(--orange); border-color: var(--orange); }
.bc-sub { font-size: 10px; color: var(--text-mut); margin: 4px 0 10px; }
/* OpenCode Go delegation-quota block (separate block inside "Token totals") */
.go-block { margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--border); }
.go-head { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-dim); margin-bottom: 5px; }
.go-row { display: flex; align-items: baseline; gap: 8px; font-size: 11px; padding: 2px 0; }
.go-name { color: var(--text-dim); min-width: 58px; text-transform: capitalize; }
.go-pct { font-weight: 600; min-width: 46px; text-align: right; }
.go-pct.go-ok { color: var(--ok-soft); }
.go-pct.go-warn { color: var(--orange); }
.go-pct.go-crit { color: var(--crit-soft); }
.go-reset { margin-left: auto; color: var(--text-dim); font-size: 10px; }
.go-unavailable { font-size: 11px; color: var(--text-mut); opacity: .75; font-style: italic; }
.bc-chart { height: 80px; margin-bottom: 10px; }
.bc-chart canvas { width: 100%; height: 80px; }
.bc-footer { display: grid; grid-template-columns: repeat(auto-fit, minmax(70px, 1fr)); gap: 8px; }
.bc-footer .ff-item .ff-label { font-size: 9px; color: var(--text-mut); text-transform: uppercase; }
.bc-footer .ff-item .ff-val { font-size: 12px; font-weight: 600; }

/* Token table */
.token-bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin: 8px 0; }
.token-bar .tb-gen { background: var(--orange); }
.token-bar .tb-cache { background: var(--blue); }
.token-bar .tb-prefill { background: var(--purple); }
.token-legend { display: flex; gap: 10px; font-size: 9px; color: var(--text-dim); margin-bottom: 8px; }
.token-legend span::before { content: '● '; }
.token-legend .tl-gen::before { color: var(--orange); }
.token-legend .tl-cache::before { color: var(--blue); }
.token-legend .tl-prefill::before { color: var(--purple); }
.token-table { width: 100%; font-size: 11px; border-collapse: collapse; }
.token-table th { text-align: left; color: var(--text-mut); font-weight: 500; padding: 4px 6px; border-bottom: 1px solid var(--border); }
.token-table td { padding: 5px 6px; border-bottom: 1px solid var(--border); }
.token-table tr.total td { font-weight: 700; border-top: 2px solid var(--border); }

/* Activity log */
.live-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 10px; color: var(--green); font-weight: 600; margin-left: auto; }
.live-badge::before { content: '●'; animation: pulse 2s infinite; }
.live-badge.paused { color: var(--text-mut); }
.live-badge.paused::before { content: '●'; animation: none; color: var(--text-mut); }
.act-toggle {
    font-size: 9px; font-weight: 700; letter-spacing: .04em;
    color: var(--text-dim); background: var(--hover-mid);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 2px 7px; cursor: pointer; margin-left: 6px; line-height: 1.4;
}
.act-toggle:hover { background: var(--hover-strong); color: var(--text); }
.act-toggle.on { color: var(--green); border-color: var(--green-border); }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.activity-list { list-style: none; max-height: 180px; overflow-y: auto; }
.activity-list li { display: flex; align-items: center; gap: 8px; padding: 6px 0; font-size: 11px; border-bottom: 1px solid var(--border); }
.activity-list li:last-child { border-bottom: none; }
.act-time { color: var(--text-mut); font-size: 10px; min-width: 28px; }
.act-dot { width: 6px; height: 6px; border-radius: 50%; }
.act-dot.green { background: var(--green); }
.act-dot.red { background: var(--red); }
.act-dot.blue { background: var(--blue); }
.act-text { color: var(--text-dim); }

/* Charts row */
.charts-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 16px; }
.chart-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px;
}
.chart-title { font-size: 11px; color: var(--text-dim); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 4px; }
.chart-legend { font-size: 10px; display: inline-flex; gap: 8px; align-items: center; }
.chart-legend .lg { font-weight: 700; }

/* Modal for Details buttons */
.modal-overlay { position: fixed; inset: 0; background: var(--overlay); display: flex; align-items: center; justify-content: center; z-index: 1000; }
.modal { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px 24px; max-width: 560px; width: 90%; max-height: 80vh; overflow-y: auto; }
.modal h3 { margin: 0 0 4px; font-size: 15px; }
.modal .modal-sub { font-size: 11px; color: var(--text-mut); margin-bottom: 14px; }
.modal table { width: 100%; border-collapse: collapse; font-size: 12px; }
.modal th, .modal td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--border); }
.modal th { color: var(--text-dim); font-weight: 600; text-transform: uppercase; font-size: 10px; }
.modal th:first-child, .modal td:first-child { text-align: left; }
.modal .modal-close { margin-top: 16px; width: 100%; background: var(--hover-mid); border: none; color: var(--text); padding: 8px; border-radius: 6px; cursor: pointer; font-size: 12px; }
.modal .modal-close:hover { background: var(--hover-strong); }
.chart-canvas { width: 100%; height: 100px; }

@media (max-width: 900px) {
    .stats-row { grid-template-columns: repeat(2, 1fr); }
    .bottom-row { grid-template-columns: 1fr; }
    .charts-row { grid-template-columns: 1fr; }
    .gpu-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
    <div class="header-left">
        <h1>Overview</h1>
        <div class="header-sub">
            <span>Mac Mini M5 Pro</span>
            <span class="badge-online" id="online-badge">1/1 online</span>
            <span>· <span id="mem-total-label">54.5 GB</span> unified memory</span>
        </div>
    </div>
    <div class="header-right">
        <div class="header-tools">
            <button class="icon-btn" id="theme-btn" type="button" title="Cambiar tema (claro/oscuro)" aria-label="Cambiar tema"><span id="theme-btn-icon">☀</span></button>
            <a class="icon-btn" href="/minions" title="Minions (subagents en vivo)" aria-label="Minions">▦</a>
            <button class="icon-btn" id="admin-btn" type="button" title="Ajustes" aria-label="Ajustes" aria-expanded="false" aria-controls="admin-panel">⚙</button>
        </div>
        <div id="header-status"></div>
    </div>
</div>

<!-- ADMIN SETTINGS PANEL (opened by #admin-btn; body is filled by per-section registries) -->
<div id="admin-panel" class="admin-panel admin-hidden" role="dialog" aria-labelledby="admin-panel-title" aria-modal="false">
    <div class="admin-panel-head">
        <div class="admin-panel-title" id="admin-panel-title">Ajustes</div>
        <button class="admin-close" id="admin-close" type="button" title="Cerrar" aria-label="Cerrar">✕</button>
    </div>
    <div class="admin-panel-body" id="admin-panel-body">
        <div class="admin-sec" id="admin-sec-bg" data-attr="bg" data-slot="bg">
            <div class="admin-sec-title">Fondo</div>
            <div class="admin-slot" id="admin-slot-bg" data-slot="bg"></div>
        </div>
        <div class="admin-sec" id="admin-sec-layout" data-attr="layout" data-slot="layout">
            <div class="admin-sec-title">Diseño</div>
            <div class="admin-slot" id="admin-slot-layout" data-slot="layout"></div>
        </div>
        <div class="admin-sec" id="admin-sec-cards" data-attr="cards" data-slot="cards">
            <div class="admin-sec-title">Tarjetas</div>
            <div class="admin-slot" id="admin-slot-cards" data-slot="cards"></div>
        </div>
        <div class="admin-sec" id="admin-sec-energy" data-attr="energy" data-slot="energy">
            <div class="admin-sec-title">Energia e historial</div>
            <div class="admin-slot" id="admin-slot-energy" data-slot="energy"></div>
        </div>
        <div class="admin-sec" id="admin-sec-refresh" data-attr="refresh" data-slot="refresh">
            <div class="admin-sec-title">Refresco</div>
            <div class="admin-slot" id="admin-slot-refresh" data-slot="refresh"></div>
        </div>
    </div>
</div>

<!-- NODE CARD -->
<div id="node-card" data-card="node"></div>

<!-- AGGREGATE STATS -->
<div class="stats-row" id="stats-row"></div>

<!-- BOTTOM ROW -->
<div class="bottom-row" id="bottom-row"></div>

<!-- CHARTS -->
<div class="charts-row" id="charts-row"></div>

<script>
// ─── Theme bootstrap: must run before anything else renders ──────────────────
// (literal key here on purpose: THEME_KEY is declared below and would be in TDZ)
(function () {
    var saved = 'dark';
    try { saved = localStorage.getItem('dashsplash.theme') || 'dark'; } catch (e) {}
    document.documentElement.setAttribute('data-theme', saved === 'light' ? 'light' : 'dark');
})();

const REFRESH_MS = 5000;             // DEFAULT poll interval (ms); runtime value lives in currentRefreshMs
// ── Refresh interval control (user-adjustable poll cadence) ─────────────────
// currentRefreshMs is the effective interval: seeded from GET /api/settings.refreshMs
// on load when valid, otherwise REFRESH_MS. Every poll re-reads it, so changing it
// takes effect on the next tick without a reload.
let currentRefreshMs = REFRESH_MS;
let refreshTimerId = null;           // pending poll timer; replaced when the interval changes
let refreshInFlight = false;         // true while a fetchData() cycle is running

function fmtRefreshMs(ms) {
    const n = Number(ms) || 0;
    if (n >= 60000 && n % 60000 === 0) return (n / 60000) + ' min';
    if (n >= 1000) return (n / 1000).toFixed(n % 1000 === 0 ? 0 : 1) + ' s';
    return n + ' ms';
}

function schedulePoll() {
    if (refreshTimerId !== null) { clearTimeout(refreshTimerId); refreshTimerId = null; }
    refreshTimerId = setTimeout(fetchData, currentRefreshMs);
    return refreshTimerId;
}

// Validate + clamp (500 ms floor keeps the dashboard from hot-looping the API).
function applyRefreshMs(ms, reschedule) {
    const n = Math.round(Number(ms));
    if (!isFinite(n) || n <= 0) return false;
    currentRefreshMs = Math.max(500, Math.min(600000, n));
    if (reschedule) schedulePoll();
    refreshUpdateHint();
    return true;
}

function refreshUpdateHint() {
    const hint = document.getElementById('refresh-hint');
    if (!hint) return;
    const preset = [1000, 3000, 5000, 10000, 30000, 60000].indexOf(currentRefreshMs) !== -1;
    hint.innerHTML = 'Intervalo actual: <b>' + fmtRefreshMs(currentRefreshMs) + '</b> (' +
        currentRefreshMs + ' ms)' + (preset ? '' : ' · personalizado');
}

function saveRefreshSetting(ms) {
    try {
        fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({refreshMs: ms}),
        }).then(r => r.json()).catch(() => {});
    } catch (e) {}
}

function loadRefreshSetting() {
    try {
        fetch('/api/settings').then(r => r.json()).then(s => {
            const ms = Number(s && s.refreshMs);
            if (isFinite(ms) && ms >= 500) {
                currentRefreshMs = Math.min(600000, Math.round(ms));
                refreshUpdateHint();
            }
        }).catch(() => {});
    } catch (e) {}
}

const history = {};
const historyKeys = ['memory_used','memory_limit','decode_tps','prefill_tps','ttft_p50','itl_p50','gpu_temp','cpu_temp','gpu_power','sys_power','gpu_util','gpu_freq','fan_rpm'];
historyKeys.forEach(k => history[k] = []);
let lastSplash = {}, lastMacmon = {}, lastOllama = {}, lastLlmstudio = {}, lastGo = {};
let activityLog = [];
let startTime = Date.now();
let lastEnergy = 0;
// ─── Energy range state (persisted history; keys match /api/energy/* ?range=) ──
const ENERGY_RANGE_META = [
    {key: '1h',  label: '1h',      unit: '1 h'},
    {key: '24h', label: '24h',     unit: '24 h'},
    {key: '7d',  label: '1sem',    unit: '1 semana'},
    {key: '30d', label: '1mes',    unit: '1 mes'},
    {key: '90d', label: '3meses',  unit: '3 meses'},
];
let energyRange = '1h';       // selected range (POSTed to /api/settings.energyRange)
let energyPoints = [];        // [{t: ms, watts, gpu}] from /api/energy/history
let energyAgg = null;         // {kwh, cost, samples, start, end} for the range
function energyRangeMeta(key) {
    return ENERGY_RANGE_META.find(r => r.key === key) || ENERGY_RANGE_META[0];
}
function energyFetch() {
    const rng = energyRange;
    return fetch('/api/energy/history?range=' + encodeURIComponent(rng))
        .then(r => r.json())
        .then(d => {
            if (!d || rng !== energyRange) return;          // stale answer, dropped
            energyPoints = (d.points || []).map(p => ({t: p.t * 1000, watts: p.watts, gpu: p.gpu_watts || 0}));
            energyAgg = {
                kwh: typeof d.kwh === 'number' ? d.kwh : 0,
                cost: typeof d.cost === 'number' ? d.cost : (d.kwh || 0) * KWH_PRICE,
                samples: d.samples || 0, start: d.start, end: d.end,
            };
            if (typeof renderBottom === 'function' && lastMacmon && Object.keys(lastMacmon).length) {
                renderBottom(lastSplash, lastMacmon, lastOllama, lastLlmstudio, lastGo);
            }
        })
        .catch(() => {});                                   // keep last good points
}
function setEnergyRange(key) {
    if (!ENERGY_RANGE_META.some(r => r.key === key)) return;
    energyRange = key;
    energyPoints = []; energyAgg = null;
    try {
        fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({energyRange: key}),
        }).catch(() => {});
    } catch (e) {}
    energyFetch();
    if (typeof renderBottom === 'function' && lastMacmon && Object.keys(lastMacmon).length) {
        renderBottom(lastSplash, lastMacmon, lastOllama, lastLlmstudio, lastGo);
    }
}
// Thinking indicator: previous /api/splash_status snapshot + active request ids
let lastSplashStatus = {};
let prevActiveIds = new Set();
let prevActiveCount = 0;
const KWH_PRICE = parseFloat("__KWH_PRICE__") || 0.13;  // € per kWh (injected from .env)

function gb(b) { return b / 1e9; }
function fmtToks(n) {
    if (n >= 1e6) return (n/1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n/1e3).toFixed(1) + 'k';
    return Math.round(n).toString();
}

function pushHistory(key, val, now) {
    history[key].push([now, val]);
    // Keep last hour
    const cutoff = Date.now() - 3600000;
    while (history[key].length > 0 && history[key][0][0] < cutoff) history[key].shift();
    // Cap at 720 points
    while (history[key].length > 720) history[key].shift();
}

function renderNode(splash, macmon, ollama, llmstudio) {
    const card = document.getElementById('node-card');
    ollama = ollama || {};
    llmstudio = llmstudio || {};
    const thermal = splash.thermal_state || 'nominal';
    const gt = macmon.gpu_temp || 0;
    const gp = macmon.gpu_power || 0;
    const gu = (macmon.gpu_util || 0) * 100;
    const gf = macmon.gpu_freq || 0;
    const ct = macmon.cpu_temp || 0;
    const sp = macmon.sys_power || 0;
    const fr = macmon.fan_rpm || 0;
    const dtps = splash.decode_tps || 0;
    const ptps = splash.prefill_tps || 0;
    const mu = splash.mem_current || 0;
    const rmTot = macmon.mem_total || 54.5e9;
    const rmUsed = macmon.mem_used || 0;
    const ollGB = gb(ollama.ram_used || 0);
    const lsGB = gb(llmstudio.ram_used || 0);   // estimated from loaded model weights
    const totGB = gb(rmTot);
    const freeGB = Math.max(0, totGB - gb(rmUsed));
    const splashGB = gb(mu);
    // segmentos de la barra (acotados a 100%)
    let sPct = totGB > 0 ? Math.min(100, splashGB / totGB * 100) : 0;
    let oPct = totGB > 0 ? Math.min(Math.max(0, 100 - sPct), ollGB / totGB * 100) : 0;
    let lPct = totGB > 0 ? Math.min(Math.max(0, 100 - sPct - oPct), lsGB / totGB * 100) : 0;
    let sysGB = Math.max(0, gb(rmUsed) - splashGB - ollGB - lsGB);
    let sysPct = totGB > 0 ? Math.min(Math.max(0, 100 - sPct - oPct - lPct), sysGB / totGB * 100) : 0;
    let freePct = Math.max(0, 100 - sPct - oPct - lPct - sysPct);
    const sd = splash.scheduler_decoding || 0;
    const sp2 = splash.scheduler_prefilling || 0;
    const sq = splash.scheduler_queued || 0;
    const running = sd + sp2;

    const tempBarPct = Math.min(gt / 95 * 100, 100);
    const powBarPct = Math.min(gp / 100 * 100, 100);
    const guBarClass = gu > 90 ? 'red' : gu > 70 ? 'orange' : 'green';

    card.innerHTML = `
    <div class="node-card">
        <div class="node-header">
            <span class="node-dot" id="node-dot"></span>
            <span class="node-name">Mac Mini M5 Pro</span>
            <span class="node-badge head">HEAD</span>
            <span class="node-ip">192.168.1.x</span>
        </div>

        <div class="mem-section">
            <div class="mem-label">Unified memory</div>
            <div class="mem-free">${freeGB.toFixed(1)} GB free</div>
            <div class="mem-bar">
                <div class="mem-bar-splash" style="width:${sPct}%"></div>
                <div class="mem-bar-ollama" style="width:${oPct}%"></div>
                <div class="mem-bar-llmstudio" style="width:${lPct}%"></div>
                <div class="mem-bar-sys" style="width:${sysPct}%"></div>
                <div class="mem-bar-free" style="width:${freePct}%"></div>
            </div>
            <div class="mem-legend">
                <span class="lg-splash">Splash ${splashGB.toFixed(1)} GB</span>
                <span class="lg-ollama">Ollama ${ollGB.toFixed(1)} GB</span>
                ${lsGB > 0.05 ? `<span class="lg-llmstudio" title="estimated from loaded weights (incl. KV-cache)">LLM Studio ${lsGB.toFixed(1)} GB</span>` : ''}
                <span class="lg-sys">System ${sysGB.toFixed(1)} GB</span>
                <span class="lg-free">Free ${freeGB.toFixed(1)} GB</span>
            </div>
        </div>

        <div class="gpu-row">
            <div class="gpu-item">
                <div class="gpu-label">GPU</div>
                <div class="gpu-val">${gu.toFixed(0)}<span class="unit">%</span></div>
                <div class="bar"><div class="bar-fill orange" style="width:${gu}%"></div></div>
            </div>
            <div class="gpu-item">
                <div class="gpu-label">GPU temp</div>
                <div class="gpu-val">${gt.toFixed(0)}<span class="unit">°C</span></div>
                <div class="bar"><div class="bar-fill green" style="width:${tempBarPct}%"></div></div>
            </div>
            <div class="gpu-item">
                <div class="gpu-label">GPU power</div>
                <div class="gpu-val">${gp.toFixed(1)} <span class="unit">/ 100 W</span></div>
                <div class="bar"><div class="bar-fill purple" style="width:${powBarPct}%"></div></div>
            </div>
            <div class="gpu-item">
                <div class="gpu-label">GPU freq</div>
                <div class="gpu-val">${gf.toFixed(0)} <span class="unit">MHz</span></div>
                <div class="bar"><div class="bar-fill blue" style="width:${Math.min(gf/2100*100,100)}%"></div></div>
            </div>
        </div>

        <div class="perf-row">
            <div class="perf-item">
                <div class="perf-label">DECODE</div>
                <div class="perf-val orange">${dtps.toFixed(1)} <span class="unit">tok/s</span></div>
            </div>
            <div class="perf-item">
                <div class="perf-label">PREFILL</div>
                <div class="perf-val purple">${ptps.toFixed(0)} <span class="unit">tok/s</span></div>
            </div>
        </div>
        <div class="perf-badge">${running} running · ${sq} queued</div>

        <div class="divider"></div>

        <div class="model-box">
            <span class="model-name">Qwen3.8-27B-4bit</span>
            <span class="model-ctx">4bit · ${splash.runtime || 'native'}</span>
            <span style="margin-left:auto;font-size:11px;color:var(--text-dim)">TTFT ${splash.ttft_p50?.toFixed(0) || '—'}ms · ITL ${splash.itl_p50?.toFixed(0) || '—'}ms</span>
            <button class="release-btn" onclick="releaseModel('splash', this)">▼ RELEASE</button>
        </div>

        ${ollama && ollama.count > 0 ? `
        <div class="divider"></div>
        <div class="model-box" style="background:var(--blue-bg);border-color:var(--blue)">
            <span style="font-size:11px;color:var(--blue);font-weight:700;text-transform:uppercase">Ollama</span>
            ${ollama.models.map(mo => `
            <span class="model-name" style="color:var(--blue)">${mo.name}</span>
            <span class="model-ctx">${(mo.ram/1e9).toFixed(1)} GB RAM${mo.vram ? ' · ' + (mo.vram/1e9).toFixed(1) + ' GB GPU' : ''}</span>
            `).join('')}
            <span style="margin-left:auto;font-size:11px;color:var(--text-dim)">${ollama.count} model${ollama.count>1?'s':''} loaded</span>
            <button class="release-btn" onclick="releaseModel('ollama', this)">▼ RELEASE</button>
        </div>` : ''}

        ${llmstudio && llmstudio.models && llmstudio.models.length > 0 ? `
        <div class="divider"></div>
        <div class="model-box" style="background:var(--orange-bg);border-color:var(--orange)">
            <span style="font-size:11px;color:var(--orange);font-weight:700;text-transform:uppercase">LLM Studio</span>
            ${llmstudio.models.map(mo => `
            <span class="model-name" style="color:var(--orange)">${mo.id}</span>
            <span class="model-ctx">${mo.tps ? mo.tps + ' t/s' : 'idle'}</span>
            `).join('')}
            <span style="margin-left:auto;font-size:11px;color:var(--text-dim)">${llmstudio.models.length} loaded</span>
            <button class="release-btn" onclick="releaseModel('llmstudio', this)">▼ RELEASE</button>
        </div>` : ''}
    </div>`;
}

function renderStats(splash, macmon) {
    const row = document.getElementById('stats-row');
    const dtps = splash.decode_tps || 0;
    const sp = macmon.sys_power || 0;
    const mu = macmon.mem_used || splash.mem_current || 0;
    const ml = macmon.mem_total || 54.5e9;
    const memPct = ml > 0 ? (mu / ml * 100) : 0;
    const genTokens = splash.decode_tokens || 0;
    const prefillTokens = splash.prefill_tokens || 0;

    row.innerHTML = `
    <div class="stat-card" data-card="stat-decode">
        <div class="stat-label">MAC DECODE</div>
        <div class="stat-value">${dtps.toFixed(1)} <span class="unit">tok/s</span></div>
        <div class="stat-spark"><canvas id="spark-decode"></canvas></div>
    </div>
    <div class="stat-card" data-card="stat-power">
        <div class="stat-label">MAC POWER</div>
        <div class="stat-value">${sp.toFixed(0)} <span class="unit">W</span></div>
        <div class="stat-spark"><canvas id="spark-power"></canvas></div>
    </div>
    <div class="stat-card" data-card="stat-memory">
        <div class="stat-label">MEMORY IN USE</div>
        <div style="display:flex;align-items:baseline;gap:6px">
            <div class="stat-value" style="color:var(--blue)">${memPct.toFixed(0)}%</div>
            <div style="font-size:13px;color:var(--text-dim)">${gb(mu).toFixed(1)} GB</div>
        </div>
        <div class="stat-bar"><div class="stat-bar-fill" style="width:${memPct}%;background:var(--blue)"></div></div>
    </div>
    <div class="stat-card" data-card="stat-tokens">
        <div class="stat-label">TOKENS TODAY</div>
        <div class="stat-value">${fmtToks(genTokens)}</div>
        <div class="stat-sub">generated · ${fmtToks(prefillTokens)} prefill</div>
    </div>`;
    // Draw sparklines
    requestAnimationFrame(() => {
        drawSpark('spark-decode', 'decode_tps', '#f9ab00');
        drawSpark('spark-power', 'sys_power', '#a144fb');
    });
}

function drawSpark(canvasId, dataKey, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.offsetWidth * 2 || 400;
    const h = canvas.height = 56;
    const pts = history[dataKey];
    if (!pts || pts.length < 2) { ctx.clearRect(0,0,w,h); return; }
    const recent = pts.slice(-30);
    const vals = recent.map(p => p[1]);
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const range = mx - mn || 1;
    ctx.clearRect(0,0,w,h);
    ctx.beginPath();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    recent.forEach((pt, i) => {
        const x = (i / (recent.length - 1)) * w;
        const y = h - 4 - ((pt[1] - mn) / range) * (h - 8);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // Fill
    ctx.lineTo((recent.length-1)/(recent.length-1)*w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = color.replace(')', ',0.08)').replace('#', '') || 'rgba(255,255,255,0.05)';
    // Use a simpler fill
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.fill();
}

function energyKWh() {
    const pts = history['sys_power'];
    if (!pts || pts.length < 2) return 0;
    let kwh = 0;
    for (let i = 1; i < pts.length; i++) {
        const dtH = (pts[i][0] - pts[i-1][0]) / 3600000;
        const p = (pts[i][1] + pts[i-1][1]) / 2;  // trapezoid avg W
        kwh += (p * dtH) / 1000;
    }
    return kwh;
}

function goResetLabel(iso) {
    try {
        if (!iso) return 'no reset info';
        const t = new Date(iso).getTime();
        if (isNaN(t)) return 'no reset info';
        const ms = t - Date.now();
        if (ms <= 0) return 'resetting now';
        const days = ms / 86400000;
        if (days >= 1) return 'resets in ' + Math.floor(days) + 'd';
        const hours = ms / 3600000;
        if (hours >= 1) return 'resets in ' + Math.floor(hours) + 'h';
        return 'resets in ' + Math.max(1, Math.floor(ms / 60000)) + 'm';
    } catch (e) {
        return 'no reset info';
    }
}

function goUsageHTML(go) {
    const head = '<div class="go-head">OpenCode Go (delegation quota)</div>';
    try {
        if (!go || !go.online) {
            const why = (go && go.reason) ? ' · ' + go.reason : '';
            return `<div class="go-block" id="go-usage-block">${head}<div class="go-unavailable">unavailable${why}</div></div>`;
        }
        const labels = { rolling: 'Rolling', weekly: 'Weekly', monthly: 'Monthly' };
        const rows = ['rolling', 'weekly', 'monthly'].map(k => {
            const w = (go.windows || {})[k] || {};
            const pct = (typeof w.percent === 'number') ? w.percent : null;
            const cls = pct === null ? '' : (pct >= 90 ? 'go-crit' : (pct >= 70 ? 'go-warn' : 'go-ok'));
            const pctTxt = pct === null ? '—' : pct.toFixed(1) + '%';
            return `<div class="go-row"><span class="go-name">${labels[k]}</span>`
                + `<span class="go-pct ${cls}">${pctTxt}</span>`
                + `<span class="go-reset">${goResetLabel(w.resetsAt)}</span></div>`;
        }).join('');
        return `<div class="go-block" id="go-usage-block">${head}${rows}</div>`;
    } catch (e) {
        return `<div class="go-block" id="go-usage-block">${head}<div class="go-unavailable">unavailable</div></div>`;
    }
}

function renderBottom(splash, macmon, ollama, llmstudio, go) {
    const row = document.getElementById('bottom-row');
    const sp = macmon.sys_power || 0;
    const gp = macmon.gpu_power || 0;
    const dtps = splash.decode_tps || 0;
    const genTokens = splash.decode_tokens || 0;
    const prefillTokens = splash.prefill_tokens || 0;
    const cachedTokens = splash.cache_reused_tokens || 0;
    const totalTokens = genTokens + cachedTokens + prefillTokens;

    // Energy: range-aware (persisted history). 1h keeps the live-session feel
    // by falling back to the in-memory trapezoid when the store has no samples.
    const eKWhLive = energyKWh();
    const rangeMeta = energyRangeMeta(energyRange);
    const hasAgg = !!(energyAgg && energyAgg.samples > 0);
    const eKWh = hasAgg ? energyAgg.kwh : (energyRange === '1h' ? eKWhLive : 0);
    const cost = hasAgg ? energyAgg.cost : eKWh * KWH_PRICE;
    const elapsedH = (Date.now() - startTime) / 3600000;
    const efficiency = genTokens > 0 ? (eKWh / genTokens * 1e6).toFixed(4) : '—';
    const rangeSub = hasAgg
        ? `${energyAgg.samples} muestras · ${rangeMeta.unit}`
        : (energyRange === '1h'
            ? 'sesión en vivo · ' + (elapsedH > 0.01 ? (elapsedH*60).toFixed(0) + ' min' : 'just started')
            : 'sin historial guardado');

    let activityHTML;
    const actItems = activityLog.slice(-8).reverse();
    if (actItems.length === 0) {
        activityHTML = '<li><span class="act-time">—</span><span class="act-dot blue"></span><span class="act-text">Waiting for events…</span></li>';
    } else {
        activityHTML = actItems.map(item => {
            const ago = Math.floor((Date.now() - item.t) / 60000);
            const timeStr = ago < 1 ? 'now' : ago < 60 ? ago + 'm' : Math.floor(ago/60) + 'h';
            return `<li><span class="act-time">${timeStr}</span><span class="act-dot ${item.color}"></span><span class="act-text">${item.text}</span></li>`;
        }).join('');
    }

    row.innerHTML = `
    <div class="bottom-card" data-card="energy">
        <div class="bc-header">
            <span class="bc-title">Energy</span>
            <span class="bc-badge">1/1 fresh</span>
            <span class="energy-ranges">${ENERGY_RANGE_META.map(r => `<button type="button" class="energy-range-btn${r.key === energyRange ? ' active' : ''}" data-range="${r.key}" onclick="setEnergyRange('${r.key}')">${r.label}</button>`).join('')}</span>
            <button class="bc-btn" style="margin-left:6px" onclick="openEnergyDetails()">Details</button>
        </div>
        <div class="bc-big">${eKWh.toFixed(2)} <span class="unit">kWh / ${rangeMeta.unit}</span></div>
        <div class="bc-sub">${cost.toFixed(2)} € · ${KWH_PRICE.toFixed(2)} €/kWh · ${rangeSub}</div>
        <div class="bc-chart"><canvas id="energy-chart"></canvas></div>
        <div class="bc-footer">
            <div class="ff-item"><div class="ff-label">Current</div><div class="ff-val">${sp.toFixed(0)} W</div></div>
            <div class="ff-item"><div class="ff-label">${rangeMeta.label}</div><div class="ff-val">${eKWh.toFixed(2)} kWh</div></div>
            <div class="ff-item"><div class="ff-label">Cost</div><div class="ff-val">${cost.toFixed(3)} €</div></div>
            <div class="ff-item"><div class="ff-label">Efficiency</div><div class="ff-val">${efficiency} kWh/k</div></div>
        </div>
    </div>

    <div class="bottom-card" data-card="tokens">
        <div class="bc-header">
            <span class="bc-title">Token totals</span>
            <span class="bc-badge">splash</span>
            <span class="bc-badge" style="margin-left:4px">All time</span>
            <button class="bc-btn" onclick="openTokenDetails()">Details</button>
        </div>
        <div class="bc-big">${fmtToks(genTokens)} <span class="unit">generated</span></div>
        <div class="bc-sub">Splash engine · Ollama has no token counters in its API</div>
        <div class="token-bar">
            <div class="tb-gen" style="width:${totalTokens > 0 ? genTokens/totalTokens*100 : 33}%"></div>
            <div class="tb-cache" style="width:${totalTokens > 0 ? cachedTokens/totalTokens*100 : 33}%"></div>
            <div class="tb-prefill" style="width:${totalTokens > 0 ? prefillTokens/totalTokens*100 : 34}%"></div>
        </div>
        <div class="token-legend">
            <span class="tl-gen">Generated</span>
            <span class="tl-cache">Cached</span>
            <span class="tl-prefill">Prefill</span>
        </div>
        <table class="token-table">
            <tr><th>MODEL</th><th>CACHED</th><th>PREFILL</th><th>GEN</th></tr>
            <tr><td>Qwen3.8-27B-4bit</td><td>${fmtToks(cachedTokens)}</td><td>${fmtToks(prefillTokens)}</td><td>${fmtToks(genTokens)}</td></tr>
            ${(llmstudio && llmstudio.models) ? llmstudio.models.filter(m => m.gen > 0 || m.tps > 0).map(mo =>
                `<tr style="color:var(--orange)"><td>LLM Studio · ${mo.id}</td><td>n/a</td><td>n/a</td><td>${fmtToks(mo.gen || 0)} (${mo.tps} t/s)</td></tr>`
            ).join('') : ''}
            ${(ollama && ollama.count > 0) ? ollama.models.map(mo =>
                `<tr style="color:var(--blue)"><td>${mo.name}</td><td>n/a</td><td>n/a</td><td>n/a</td></tr>`
            ).join('') : ''}
            <tr class="total"><td>TOTAL (splash)</td><td>${fmtToks(cachedTokens)}</td><td>${fmtToks(prefillTokens)}</td><td>${fmtToks(genTokens)}</td></tr>
        </table>
        ${(ollama && ollama.count > 0) ? `<div class="bc-sub" style="margin-top:8px">Ollama: ${ollama.count} model${ollama.count>1?'s':''} loaded · ${(ollama.ram_used/1e9).toFixed(1)} GB · tokens n/a (no API counter)</div>` : ''}
        ${goUsageHTML(go)}
    </div>

    <div class="bottom-card" data-card="activity">
        <div class="bc-header">
            <span class="bc-title">Recent activity</span>
            <span class="live-badge" id="act-badge">LIVE</span>
            <button class="act-toggle on" id="act-toggle" onclick="toggleActivityPanel()" title="Show / hide the activity feed">ON</button>
        </div>
        <ul class="activity-list" id="activity-list">${activityHTML}</ul>
        <button class="bc-btn" id="act-viewall" style="margin-top:8px;margin-left:0" onclick="openActivityDetails()">View all activity</button>
    </div>`;

    // Draw energy chart (synchronously)
    drawEnergyChart();
    applyActivityVisibility();
    setTimeout(() => renderActivity(), 0);
}

function drawEnergyChart() {
    const canvas = document.getElementById('energy-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.offsetWidth * 2 || 400;
    const h = canvas.height = 160;
    ctx.clearRect(0, 0, w, h);
    // Prefer the persisted range history; fall back to the live session series
    // (1h / default view, or when /api/energy/history has nothing yet).
    let pts;
    if (energyPoints.length >= 2) {
        pts = energyPoints.map(p => [p.t, p.watts, p.gpu]);
    } else {
        pts = (history['sys_power'] || []).slice(-60).map(p => [p[0], p[1], 0]);
    }
    if (!pts || pts.length < 2) return;
    const mx = Math.max(...pts.map(p => p[1]), 10);
    const pad = {t: 6, r: 6, b: 18, l: 6};
    const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
    const step = cw / pts.length;
    const barW = Math.max(1, step - 1);
    // Watts bars
    pts.forEach((pt, i) => {
        const x = pad.l + i * step;
        const bh = Math.max(1, (pt[1] / mx) * ch);
        ctx.fillStyle = pt[1] > mx * 0.8 ? 'rgba(234,67,53,0.7)' : 'rgba(249,171,0,0.6)';
        ctx.fillRect(x, pad.t + ch - bh, barW, bh);
    });
    // GPU watts overlay (line)
    if (pts.some(p => p[2] > 0)) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(161,68,251,0.9)';
        ctx.lineWidth = 2;
        pts.forEach((pt, i) => {
            const x = pad.l + i * step + step / 2;
            const y = pad.t + ch - Math.min(1, pt[2] / mx) * ch;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
    // Time labels: dates for multi-day ranges, HH:MM for short ones
    const spanMs = pts[pts.length - 1][0] - pts[0][0];
    const fmt = ms => {
        const d = new Date(ms);
        return spanMs > 2 * 86400000
            ? (d.getMonth() + 1) + '/' + d.getDate()
            : String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
    };
    ctx.fillStyle = '#5f6368'; ctx.font = '16px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText(fmt(pts[0][0]), pad.l, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(fmt(pts[pts.length - 1][0]), w - pad.r, h - 4);
}

function renderActivity() {
    const list = document.getElementById('activity-list');
    if (!list) return;
    const items = activityLog.slice(-8).reverse();
    if (items.length === 0) {
        list.innerHTML = '<li><span class="act-time">—</span><span class="act-dot blue"></span><span class="act-text">Waiting for events…</span></li>';
        return;
    }
    list.innerHTML = items.map(item => {
        const ago = Math.floor((Date.now() - item.t) / 60000);
        const timeStr = ago < 1 ? 'now' : ago < 60 ? ago + 'm' : Math.floor(ago/60) + 'h';
        return `<li><span class="act-time">${timeStr}</span><span class="act-dot ${item.color}"></span><span class="act-text">${item.text}</span></li>`;
    }).join('');
}

/* ─── Activity panel on/off toggle (persisted in localStorage) ───────────── */
const ACT_TOGGLE_KEY = 'dashsplash.activityOn';
function activityEnabled() {
    try { return localStorage.getItem(ACT_TOGGLE_KEY) !== '0'; } catch (e) { return true; }
}
function applyActivityVisibility() {
    const on = activityEnabled();
    const list = document.getElementById('activity-list');
    const btn  = document.getElementById('act-viewall');
    const badge = document.getElementById('act-badge');
    const tog  = document.getElementById('act-toggle');
    if (list) list.style.display = on ? '' : 'none';
    if (btn)  btn.style.display  = on ? '' : 'none';
    if (badge) {
        badge.classList.toggle('paused', !on);
        badge.textContent = on ? 'LIVE' : 'PAUSED';
    }
    if (tog) {
        tog.classList.toggle('on', on);
        tog.textContent = on ? 'ON' : 'OFF';
        tog.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
}
function toggleActivityPanel() {
    try { localStorage.setItem(ACT_TOGGLE_KEY, activityEnabled() ? '0' : '1'); } catch (e) {}
    applyActivityVisibility();
}

function renderCharts() {
    const row = document.getElementById('charts-row');
    row.innerHTML = `
        <div class="chart-card" data-card="chart-temp">
            <div class="chart-title"><span>GPU temperature</span>
                <span class="chart-legend"><span class="lg" style="color:#ea4335">GPU</span><span class="lg" style="color:#f9ab00">CPU</span> <span id="ct-gpu">—</span></span>
            </div>
            <canvas id="chart-temp" class="chart-canvas"></canvas>
        </div>
        <div class="chart-card" data-card="chart-power">
            <div class="chart-title"><span>Power</span>
                <span class="chart-legend"><span class="lg" style="color:#a144fb">GPU</span><span class="lg" style="color:#4285f4">System</span> <span id="ct-power">—</span></span>
            </div>
            <canvas id="chart-power" class="chart-canvas"></canvas>
        </div>
        <div class="chart-card" data-card="chart-throughput">
            <div class="chart-title"><span>Throughput</span>
                <span class="chart-legend"><span class="lg" style="color:#34a853">Decode</span><span class="lg" style="color:#a144fb">Prefill</span> <span id="ct-tp">—</span></span>
            </div>
            <canvas id="chart-tp" class="chart-canvas"></canvas>
        </div>`;
    requestAnimationFrame(() => {
        drawLineChart('chart-temp', ['gpu_temp','cpu_temp'], ['#ea4335','#f9ab00'], 30, 100, '°C', 'ct-gpu');
        drawLineChart('chart-power', ['gpu_power','sys_power'], ['#a144fb','#4285f4'], 0, 100, 'W', 'ct-power');
        drawLineChart('chart-tp', ['decode_tps','prefill_tps'], ['#34a853','#a144fb'], 0, 500, 'tok/s', 'ct-tp');
    });
}

function drawLineChart(canvasId, keys, colors, yMin, yMax, unit, labelId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.offsetWidth * 2 || 500;
    const h = canvas.height = 200;
    const pad = {t:10, r:10, b:20, l:50};
    const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;

    ctx.clearRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = '#2a2f3a'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = pad.t + (ch / 4) * i;
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
        const val = yMax - (yMax - yMin) * (i / 4);
        ctx.fillStyle = '#5f6368'; ctx.font = '14px system-ui'; ctx.textAlign = 'right';
        ctx.fillText(unit === '°C' ? val.toFixed(0)+'°' : val.toFixed(0), pad.l - 5, y + 4);
    }

    let allData = [];
    keys.forEach((k, ki) => {
        const pts = (history[k] || []).slice(-60);
        allData.push(pts);
    });

    const maxLen = Math.max(...allData.map(d => d.length), 1);
    // Update label
    const labelEl = document.getElementById(labelId);
    if (labelEl) {
        const lastVals = keys.map((k, ki) => {
            const pts = allData[ki];
            return pts.length > 0 ? pts[pts.length-1][1].toFixed(0) + (unit==='°C'?'°':unit==='W'?'W':'') : '—';
        });
        labelEl.textContent = lastVals.join('  ');
    }

    allData.forEach((pts, ki) => {
        if (pts.length < 1) return;
        ctx.beginPath();
        ctx.strokeStyle = colors[ki]; ctx.lineWidth = 2; ctx.lineJoin = 'round';
        pts.forEach((pt, i) => {
            const x = pad.l + (i / Math.max(pts.length - 1, 1)) * cw;
            const clampedY = Math.max(yMin, Math.min(yMax, pt[1]));
            const y = pad.t + ch - ((clampedY - yMin) / (yMax - yMin)) * ch;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        // Last point dot
        const last = pts[pts.length - 1];
        const lx = pad.l + ((pts.length - 1) / Math.max(pts.length - 1, 1)) * cw;
        const clampedLy = Math.max(yMin, Math.min(yMax, last[1]));
        const ly = pad.t + ch - ((clampedLy - yMin) / (yMax - yMin)) * ch;
        ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2);
        ctx.fillStyle = colors[ki]; ctx.fill();
    });
}

function logActivity(text, color) {
    activityLog.push({t: Date.now(), text, color: color || 'green'});
    if (activityLog.length > 50) activityLog.shift();
}

function updateActivity(splash, macmon, status) {
    const prevThermal = lastSplash.thermal_state;
    const curThermal = splash.thermal_state || 'nominal';
    if (prevThermal && curThermal !== prevThermal) {
        logActivity(`Thermal: ${prevThermal} → ${curThermal}`, curThermal === 'critical' ? 'red' : 'blue');
    }
    if (lastSplash.scheduler_decoding === 0 && splash.scheduler_decoding > 0) {
        logActivity('Decode started', 'green');
    }
    if (lastSplash.scheduler_decoding > 0 && splash.scheduler_decoding === 0) {
        logActivity('Decode finished', 'blue');
    }
    if (lastMacmon.gpu_temp < 80 && (macmon.gpu_temp || 0) >= 80) {
        logActivity(`GPU hot: ${macmon.gpu_temp.toFixed(0)}°C`, 'red');
    }
    const d = splash.splash_delta;
    if (d) {
        if (d.completed > 0) {
            const inFmt = n => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
            let txt = `Splash: ${d.completed} req · in ${inFmt(d.input + d.cached)} · cached ${inFmt(d.cached)} · out ${inFmt(d.output)}`;
            const tps = splash.decode_tps || 0;
            if (tps > 0) txt += ` · ${tps.toFixed(1)} tok/s`;
            const ttft = splash.ttft_p50 || 0;
            if (ttft > 0) txt += ` · TTFT ${ttft < 1000 ? ttft.toFixed(0) + 'ms' : (ttft / 1000).toFixed(1) + 's'}`;
            const fail = (splash.requests_failed || 0) - (lastSplash.requests_failed || 0);
            if (fail > 0) { txt += ` · ${fail} failed`; }
            logActivity(txt, fail > 0 ? 'orange' : 'green');
        }
    }

    // ── Thinking indicator (edge-triggered, from /api/splash_status) ──────
    // Only log when a request ENTERS the active set (idle -> active, or a new
    // id shows up) or when the set drains back to idle. Never per-poll.
    if (status && status.ok) {
        lastSplashStatus = status;
        const reqs = Array.isArray(status.active_requests) ? status.active_requests : [];
        const curIds = new Set(reqs.map(r => String(r.id)));
        const newReqs = reqs.filter(r => !prevActiveIds.has(String(r.id)));
        const thinking = status.reasoning_switch ? 'Thinking on' : 'Request active';
        if (reqs.length > 0 && (prevActiveCount === 0 || newReqs.length > 0)) {
            const show = newReqs.length ? newReqs : reqs.slice(0, 1);
            const r = show[0];
            const tok = (r.phase === 'prefill')
                ? (r.prompt_processed || r.prompt_tokens || 0)
                : (r.generated_tokens || 0);
            const unit = r.phase === 'prefill' ? 'tok ctx' : 'tok';
            let txt = `${thinking} - ${r.phase} - ${tok} ${unit}`;
            if (reqs.length > 1) txt += ` (+${reqs.length - 1} more)`;
            logActivity(txt, 'blue');
        } else if (reqs.length === 0 && prevActiveCount > 0) {
            logActivity('idle', 'blue');
        }
        prevActiveIds = curIds;
        prevActiveCount = reqs.length;
    }
}

function updateStatus(splash, macmon, ollama, llmstudio) {
    const splashOk = splash.decode_tps !== undefined || splash.mem_current !== undefined;
    const macmonOk = macmon.gpu_temp !== undefined;
    const ollamaOk = !!(ollama && ollama.models !== undefined);
    const llmstudioOk = !!(llmstudio && llmstudio.online);
    const el = document.getElementById('header-status');
    const now = new Date().toLocaleTimeString();
    const thermal = splash.thermal_state || 'nominal';
    const ollamaState = (ollama && ollama.count > 0) ? 'OK' : 'idle';
    const llmstudioState = (llmstudio && Array.isArray(llmstudio.models) && llmstudio.models.length > 0) ? 'OK' : 'idle';
    const sc = s => s === 'OK' ? 'var(--green)' : s === 'idle' ? 'var(--orange)' : 'var(--red)';
    el.innerHTML = `Splash: <span style="color:${sc(splashOk?'OK':'FAIL')}">${splashOk?'OK':'FAIL'}</span>
        &nbsp; Ollama: <span style="color:${sc(ollamaOk?ollamaState:'FAIL')}">${ollamaOk?ollamaState:'FAIL'}</span>
        &nbsp; LLM Studio: <span style="color:${sc(llmstudioOk?llmstudioState:'FAIL')}">${llmstudioOk?llmstudioState:'FAIL'}</span>
        &nbsp; Macmon: <span style="color:${macmonOk?'var(--green)':'var(--red)'}">${macmonOk?'OK':'FAIL'}</span>
        &nbsp; Thermal: <span style="color:${thermal==='critical'?'var(--red)':thermal==='serious'?'var(--orange)':thermal==='fair'?'var(--orange)':'var(--green)'}">${thermal}</span>
        &nbsp; Runtime: ${splash.runtime || 'native'}
        &nbsp; <span style="color:var(--text-mut)">${now}</span>`;
}

function releaseModel(engine, btn) {
    if (!btn) return;
    btn.disabled = true;
    const orig = btn.textContent;
    btn.textContent = '…';
    fetch('/api/release', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({engine: engine}),
    }).then(r => r.json()).then(res => {
        btn.textContent = res.ok ? '✓ released' : '✗ ' + (res.status || res.error || 'failed');
        if (res.models && res.models.length) {
            btn.textContent = res.models.every(m => m.ok) ? '✓ all released' : '✗ ' + res.models.filter(m=>!m.ok).map(m=>m.model).join(', ');
        }
    }).catch(() => { btn.textContent = '✗ error'; })
    .finally(() => {
        setTimeout(() => { btn.disabled = false; btn.textContent = orig; }, 2500);
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// ─── Admin and theme ────────────────────────────────────────────────────────
// Theme: html[data-theme] = 'dark' (default) | 'light', persisted in localStorage.
// Panel: #admin-panel is a shell. Other modules register their own UI with
//        window.adminSections[slot] = function () { ...inject into [data-slot=slot]... }
//        slots: bg | layout | cards | energy | refresh
// ═══════════════════════════════════════════════════════════════════════════
const THEME_KEY = 'dashsplash.theme';
const ADMIN_SLOTS = ['bg', 'layout', 'cards', 'energy', 'refresh'];
window.adminSections = window.adminSections || {};

function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

function applyTheme(theme) {
    const t = (theme === 'light') ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', t);
    try { localStorage.setItem(THEME_KEY, t); } catch (e) { /* private mode */ }
    const icon = document.getElementById('theme-btn-icon');
    if (icon) icon.textContent = (t === 'light') ? '☾' : '☀';
    // Charts are canvas: tell any redrawer to refresh so it can re-read tokens.
    try { if (typeof window.onThemeChange === 'function') window.onThemeChange(t); } catch (e) {}
    return t;
}

function saveThemeSetting(theme) {
    try {
        fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({theme: theme}),
        }).catch(() => {});
    } catch (e) {}
}

function toggleTheme() {
    const next = currentTheme() === 'light' ? 'dark' : 'light';
    applyTheme(next);
    saveThemeSetting(next);
}

// ── Custom background image (admin slot "bg") ──────────────────────────────
// Persisted filename lives in SETTINGS.bgFile (server-side). The photo is
// applied to <body> via the --bg-image custom property; <body> survives every
// dashboard re-render, so the background is NOT touched by fetchData().
var bgCurrentFile = null;

function bgApply(filename) {
    bgCurrentFile = filename || null;
    var b = document.body;
    if (!b) return;
    if (bgCurrentFile) {
        b.classList.add('has-bg');
        b.style.setProperty('--bg-image', "url('/bg/" + encodeURIComponent(bgCurrentFile) + "')");
    } else {
        b.classList.remove('has-bg');
        b.style.removeProperty('--bg-image');
    }
}

function bgPersist(filename) {
    fetch('/api/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({bgFile: filename || null}),
    }).catch(function () {});
}

function bgSet(filename) {          // apply now + persist
    bgApply(filename);
    bgPersist(filename);
    bgRefreshSlot();
}

function bgRefreshSlot() {
    var wrap = document.querySelector('#admin-sec-bg .admin-slot #bg-preview-wrap');
    if (!wrap) return;
    wrap.innerHTML = '';
    var thumb = document.createElement('div');
    thumb.className = 'bg-thumb';
    if (bgCurrentFile) {
        var img = document.createElement('img');
        img.src = '/bg/' + encodeURIComponent(bgCurrentFile) + '?t=' + Date.now();
        img.alt = 'Fondo actual';
        img.style.cssText = 'display:block;width:100%;height:96px;object-fit:cover;border-radius:8px;border:1px solid var(--border);margin:6px 0 8px;';
        wrap.appendChild(img);
    } else {
        thumb.classList.add('bg-empty');
        thumb.textContent = 'Sin imagen de fondo';
        wrap.appendChild(thumb);
    }
}

window.adminSections['bg'] = function () {
    var slot = document.querySelector('#admin-sec-bg .admin-slot') ||
               document.querySelector('.admin-slot[data-slot="bg"]');
    if (!slot) return;
    if (slot.dataset.bgWired === '1') { bgRefreshSlot(); return; }
    slot.dataset.bgWired = '1';
    slot.innerHTML =
        '<div class="bg-preview" id="bg-preview-wrap"></div>' +
        '<div class="bg-row">' +
          '<label class="bg-file-label">Subir foto' +
            '<input type="file" id="bg-file-input" accept="image/*">' +
          '</label>' +
          '<button type="button" id="bg-clear-btn">Quitar</button>' +
        '</div>' +
        '<div class="bg-status" id="bg-status"></div>';

    bgRefreshSlot();

    var input = slot.querySelector('#bg-file-input');
    var status = slot.querySelector('#bg-status');

    input.addEventListener('change', function () {
        var f = input.files && input.files[0];
        if (!f) return;
        if (f.type.indexOf('image/') !== 0) { status.textContent = '✗ no es una imagen'; return; }
        status.textContent = 'Subiendo…';
        var reader = new FileReader();
        reader.onload = function () {
            fetch('/api/bg', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image: reader.result, name: f.name}),
            }).then(function (r) { return r.json(); })
              .then(function (d) {
                  if (d && d.ok && d.filename) {
                      bgSet(d.filename);
                      status.textContent = '✓ ' + d.filename;
                  } else {
                      status.textContent = '✗ ' + ((d && d.error) || 'error');
                  }
              })
              .catch(function () { status.textContent = '✗ error de red'; })
              .finally(function () { input.value = ''; });
        };
        reader.onerror = function () { status.textContent = '✗ no se pudo leer el fichero'; };
        reader.readAsDataURL(f);
    });

    slot.querySelector('#bg-clear-btn').addEventListener('click', function () {
        status.textContent = 'Quitando…';
        fetch('/api/bg', {method: 'POST', headers: {'Content-Type': 'application/json'},
                          body: JSON.stringify({clear: true})})
          .then(function (r) { return r.json(); })
          .then(function (d) {
              bgSet(null);
              status.textContent = (d && d.ok) ? '✓ fondo eliminado' : '✗ error';
          })
          .catch(function () { bgSet(null); status.textContent = '✓ fondo quitado (local)'; });
    });
};

// Apply the persisted background as soon as the DOM is ready.
function bgInitFromServer() {
    fetch('/api/bg').then(function (r) { return r.json(); })
      .then(function (d) { bgApply(d && d.filename ? d.filename : null); })
      .catch(function () { /* keep default theme background */ });
}
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bgInitFromServer);
} else {
    bgInitFromServer();
}

// ── Admin panel ────────────────────────────────────────────────────────────
function adminRender(slot) {
    try {
        const fn = window.adminSections[slot];
        if (typeof fn === 'function') fn();
    } catch (e) {
        console.warn('[admin] section "' + slot + '" failed:', e);
    }
}

function adminRenderAll() { ADMIN_SLOTS.forEach(adminRender); }

function adminIsOpen() {
    const p = document.getElementById('admin-panel');
    return !!p && !p.classList.contains('admin-hidden');
}

function adminOpen() {
    const p = document.getElementById('admin-panel');
    if (!p) return;
    p.classList.remove('admin-hidden');
    const btn = document.getElementById('admin-btn');
    if (btn) { btn.classList.add('active'); btn.setAttribute('aria-expanded', 'true'); }
    adminRenderAll();
}

function closeAdmin() {
    const p = document.getElementById('admin-panel');
    if (!p) return;
    p.classList.add('admin-hidden');
    const btn = document.getElementById('admin-btn');
    if (btn) { btn.classList.remove('active'); btn.setAttribute('aria-expanded', 'false'); }
}

function adminToggle() { adminIsOpen() ? closeAdmin() : adminOpen(); }

(function initAdminAndTheme() {
    const themeBtn = document.getElementById('theme-btn');
    if (themeBtn) themeBtn.addEventListener('click', function (e) { e.stopPropagation(); toggleTheme(); });

    const adminBtn = document.getElementById('admin-btn');
    if (adminBtn) adminBtn.addEventListener('click', function (e) { e.stopPropagation(); adminToggle(); });

    const closeBtn = document.getElementById('admin-close');
    if (closeBtn) closeBtn.addEventListener('click', function (e) { e.stopPropagation(); closeAdmin(); });

    // Close on outside click (clicks inside the panel stay inside).
    document.addEventListener('click', function (e) {
        if (!adminIsOpen()) return;
        const p = document.getElementById('admin-panel');
        if (p && p.contains(e.target)) return;
        if (adminBtn && adminBtn.contains(e.target)) return;
        closeAdmin();
    });

    // Close on Escape.
    document.addEventListener('keydown', function (e) {
        if (e.key === 'Escape' && adminIsOpen()) closeAdmin();
    });

    // Sync the theme icon with whatever the bootstrap applied.
    applyTheme(currentTheme() === 'light' ? 'light' : 'dark');
})();

// ── Refresh section: presets + custom seconds + live current-interval hint ──
// Reads/writes currentRefreshMs (see top of script). Persists refreshMs through
// POST /api/settings so the choice survives a dashboard reload.
(function initRefreshSection() {
    const PRESETS = [
        {ms: 1000, label: '1 s'},
        {ms: 3000, label: '3 s'},
        {ms: 5000, label: '5 s'},
        {ms: 10000, label: '10 s'},
        {ms: 30000, label: '30 s'},
        {ms: 60000, label: '1 min'},
    ];

    function injectCss() {
        if (document.getElementById('refresh-css')) return;
        const st = document.createElement('style');
        st.id = 'refresh-css';
        st.textContent = [
            '#admin-sec-refresh .rf-grid { display:flex; flex-wrap:wrap; gap:6px; }',
            '#admin-sec-refresh .rf-btn { font-weight:600; min-width:52px; padding:5px 8px; }',
            '#admin-sec-refresh .rf-btn.active { background:var(--green-bg); color:var(--green);',
            '  border-color:var(--green); font-weight:700; }',
            '#admin-sec-refresh .rf-custom { display:flex; align-items:center; gap:6px; margin-top:10px; }',
            '#admin-sec-refresh .rf-custom input { width:70px; }',
            '#admin-sec-refresh .rf-custom label { padding:0; }',
            '#admin-sec-refresh .rf-hint { margin-top:9px; font-size:11px; color:var(--text-mut); }',
            '#admin-sec-refresh .rf-hint b { color:var(--text); }',
        ].join('');
        document.head.appendChild(st);
    }

    function paint(root) {
        root.querySelectorAll('.rf-btn').forEach(b => {
            b.classList.toggle('active', Number(b.dataset.ms) === currentRefreshMs);
            b.setAttribute('aria-pressed', Number(b.dataset.ms) === currentRefreshMs ? 'true' : 'false');
        });
        const inp = root.querySelector('#rf-custom-input');
        if (inp && document.activeElement !== inp) inp.value = String(currentRefreshMs / 1000);
        refreshUpdateHint();
    }

    function choose(root, ms) {
        if (!applyRefreshMs(ms, true)) { paint(root); return; }
        saveRefreshSetting(currentRefreshMs);
        paint(root);
    }

    window.adminSections['refresh'] = function () {
        injectCss();
        const root = document.querySelector('#admin-sec-refresh .admin-slot') ||
                     document.getElementById('admin-slot-refresh');
        if (!root) return;

        if (!root.dataset.built) {
            root.dataset.built = '1';
            root.innerHTML =
                '<div class="rf-grid" role="group" aria-label="Intervalo de refresco"></div>' +
                '<div class="rf-custom">' +
                    '<label for="rf-custom-input">Personalizado</label>' +
                    '<input id="rf-custom-input" type="number" min="1" max="600" step="1" ' +
                        'inputmode="numeric" aria-label="Intervalo personalizado en segundos">' +
                    '<span>s</span>' +
                    '<button type="button" class="rf-apply">Aplicar</button>' +
                '</div>' +
                '<div class="rf-hint" id="refresh-hint" aria-live="polite"></div>';

            const grid = root.querySelector('.rf-grid');
            PRESETS.forEach(p => {
                const b = document.createElement('button');
                b.type = 'button';
                b.className = 'rf-btn';
                b.dataset.ms = String(p.ms);
                b.textContent = p.label;
                b.addEventListener('click', function (e) { e.stopPropagation(); choose(root, p.ms); });
                grid.appendChild(b);
            });

            const inp = root.querySelector('#rf-custom-input');
            const applyCustom = function () {
                const secs = Number(inp.value);
                if (!isFinite(secs) || secs <= 0) { paint(root); return; }
                const ms = Math.max(500, Math.min(600000, Math.round(secs * 1000)));
                inp.value = String(ms / 1000);
                choose(root, ms);
            };
            root.querySelector('.rf-apply').addEventListener('click', function (e) {
                e.stopPropagation(); applyCustom();
            });
            inp.addEventListener('keydown', function (e) {
                if (e.key === 'Enter') { e.preventDefault(); applyCustom(); }
            });
            inp.addEventListener('change', applyCustom);
        }
        paint(root);
    };

    // Seed the effective interval from the persisted setting (default = REFRESH_MS).
    loadRefreshSetting();
})();

function fetchData() {
    if (refreshInFlight) return;      // single-flight: one poll chain only
    refreshInFlight = true;
    Promise.all([
        fetch('/api/splash').then(r => r.json()).catch(() => ({})),
        fetch('/api/macmon').then(r => r.json()).catch(() => ({})),
        fetch('/api/ollama').then(r => r.json()).catch(() => ({})),
        fetch('/api/llmstudio').then(r => r.json()).catch(() => ({})),
        fetch('/api/opencodego').then(r => r.json()).catch(() => ({})),
        fetch('/api/splash_status').then(r => r.json()).catch(() => ({})),
    ]).then(([splash, macmon, ollama, llmstudio, opencodego, splash_status]) => {
        const now = Date.now();
        lastSplash = splash;
        lastMacmon = macmon;
        lastOllama = ollama;
        lastLlmstudio = llmstudio;
        lastGo = opencodego;

        if (splash.decode_tps !== undefined) {
            pushHistory('memory_used', splash.mem_current / 1e9, now);
            pushHistory('memory_limit', splash.mem_limit / 1e9, now);
            pushHistory('decode_tps', splash.decode_tps, now);
            pushHistory('prefill_tps', splash.prefill_tps, now);
            pushHistory('ttft_p50', splash.ttft_p50, now);
            pushHistory('itl_p50', splash.itl_p50, now);
        }
        if (macmon.gpu_temp !== undefined) {
            pushHistory('gpu_temp', macmon.gpu_temp, now);
            pushHistory('cpu_temp', macmon.cpu_temp, now);
            pushHistory('gpu_power', macmon.gpu_power, now);
            pushHistory('sys_power', macmon.sys_power, now);
            pushHistory('gpu_util', macmon.gpu_util, now);
            pushHistory('gpu_freq', macmon.gpu_freq, now);
            pushHistory('fan_rpm', macmon.fan_rpm, now);
        }

        updateActivity(splash, macmon, splash_status);
        renderNode(splash, macmon, ollama, llmstudio);
        renderStats(splash, macmon);
        renderBottom(splash, macmon, ollama, llmstudio, opencodego);
        renderCharts();
        applyCardLayout();
        updateStatus(splash, macmon, ollama, llmstudio);

        refreshInFlight = false;
        schedulePoll();
    }).catch(() => {
        refreshInFlight = false;
        schedulePoll();
    });
}

// Initial load
fetchData();

// ═══════════════════════════════════════════════════════════════════════════
// ─── Energy range boot + admin section (energy slot) ────────────────────────
// The selected range is persisted in /api/settings.energyRange; the 1h view
// falls back to the live session series so it behaves as before.
// ═══════════════════════════════════════════════════════════════════════════
(function initEnergyRange() {
    fetch('/api/settings').then(r => r.json()).then(s => {
        if (s && ENERGY_RANGE_META.some(r => r.key === s.energyRange)) energyRange = s.energyRange;
    }).catch(() => {}).then(() => energyFetch());
    setInterval(energyFetch, 60000);        // keep the range chart fresh
})();

window.adminSections['energy'] = function () {
    const el = document.querySelector('#admin-sec-energy .admin-slot')
        || document.getElementById('admin-slot-energy');
    if (!el) return;
    const selId = 'energy-default-range';
    el.innerHTML = `
        <label>Rango por defecto
            <select id="${selId}">
                ${ENERGY_RANGE_META.map(r => `<option value="${r.key}"${r.key === energyRange ? ' selected' : ''}>${r.unit}</option>`).join('')}
            </select>
        </label>
        <div style="margin-top:10px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
            <button id="energy-clear-btn" type="button">Borrar historial</button>
            <span id="energy-clear-msg" style="font-size:11px;color:var(--text-mut)"></span>
        </div>
        <div id="energy-hist-info" style="margin-top:6px;font-size:11px;color:var(--text-mut)">cargando…</div>`;

    const sel = document.getElementById(selId);
    if (sel) sel.addEventListener('change', () => setEnergyRange(sel.value));

    const clr = document.getElementById('energy-clear-btn');
    if (clr) clr.addEventListener('click', () => {
        if (!confirm('¿Borrar todo el historial de consumo guardado?')) return;
        const msg = document.getElementById('energy-clear-msg');
        clr.disabled = true;
        fetch('/api/energy/clear', {method: 'POST'})
            .then(r => r.json())
            .then(res => {
                if (msg) msg.textContent = (res && res.ok) ? `borrado (${res.removed} muestras)` : 'error';
                energyPoints = []; energyAgg = null;
                energyFetch();
                const info = document.getElementById('energy-hist-info');
                if (info) info.textContent = 'sin historial guardado todavía';
            })
            .catch(() => { if (msg) msg.textContent = 'error'; })
            .then(() => { clr.disabled = false; });
    });

    fetch('/api/energy/summary?range=90d').then(r => r.json()).then(d => {
        const info = document.getElementById('energy-hist-info');
        if (!info || !d) return;
        const n = d.samples || 0;
        if (!n) { info.textContent = 'sin historial guardado todavía'; return; }
        const from = d.start ? new Date(d.start * 1000).toLocaleDateString() : '—';
        info.textContent = `${n} muestras · desde ${from} · ${(d.kwh || 0).toFixed(2)} kWh / 90d · ${(d.cost || 0).toFixed(2)} €`;
    }).catch(() => {});
};


// ═══════════════════════════════════════════════════════════════════════════
// ─── Card layout & visibility ───────────────────────────────────────────────
// Single source of truth: window.dashPrefs = {visible:{id:bool}, order:[id...],
// mode:'auto'|'onecol'|'twocol'}.  applyCardLayout() is idempotent and runs at
// the END of every render pass (plus after every admin change), so visibility,
// order and column mode survive the periodic innerHTML re-render of the rows.
// ───────────────────────────────────────────────────────────────────────────
const CARD_NATURAL_ORDER = ['node','stat-decode','stat-power','stat-memory','stat-tokens',
    'energy','tokens','activity','chart-temp','chart-power','chart-throughput'];
const CARD_LABELS = {
    'node': 'Nodo / GPU',
    'stat-decode': 'Stats · MAC decode',
    'stat-power': 'Stats · MAC power',
    'stat-memory': 'Stats · memoria',
    'stat-tokens': 'Stats · tokens hoy',
    'energy': 'Energía',
    'tokens': 'Tokens totales',
    'activity': 'Actividad reciente',
    'chart-temp': 'Gráfico · temperatura',
    'chart-power': 'Gráfico · potencia',
    'chart-throughput': 'Gráfico · throughput'
};
const CARD_ROWS = ['stats-row', 'bottom-row', 'charts-row'];

window.dashPrefs = window.dashPrefs || { visible: {}, order: [], mode: 'auto' };
if (!window.dashPrefs.visible || typeof window.dashPrefs.visible !== 'object') window.dashPrefs.visible = {};
if (!Array.isArray(window.dashPrefs.order)) window.dashPrefs.order = [];
if (['auto', 'onecol', 'twocol'].indexOf(window.dashPrefs.mode) < 0) window.dashPrefs.mode = 'auto';

function cardIdsInDom() {
    const out = [];
    document.querySelectorAll('[data-card]').forEach(function (el) {
        const id = el.getAttribute('data-card');
        if (id && out.indexOf(id) < 0) out.push(id);
    });
    return out;
}

// persisted order (may be partial / stale) + every card currently in the DOM
function cardOrderFull() {
    const dom = cardIdsInDom();
    const out = [];
    (window.dashPrefs.order || []).forEach(function (id) {
        if (dom.indexOf(id) >= 0 && out.indexOf(id) < 0) out.push(id);
    });
    dom.forEach(function (id) { if (out.indexOf(id) < 0) out.push(id); });
    return out;
}

function applyCardLayout() {
    const prefs = window.dashPrefs;
    const visible = prefs.visible || {};
    const order = prefs.order || [];
    const mode = ['auto', 'onecol', 'twocol'].indexOf(prefs.mode) >= 0 ? prefs.mode : 'auto';

    function rank(id) { const i = order.indexOf(id); return i < 0 ? 1e6 : i; }

    // 1) visibility
    document.querySelectorAll('[data-card]').forEach(function (el) {
        const id = el.getAttribute('data-card');
        el.style.display = (visible[id] === false) ? 'none' : '';
    });

    // 2) order inside each row + 3) column mode
    CARD_ROWS.forEach(function (rowId) {
        const row = document.getElementById(rowId);
        if (!row) return;
        row.setAttribute('data-mode', mode);
        const kids = Array.prototype.slice.call(row.children).filter(function (el) {
            return el.hasAttribute && el.hasAttribute('data-card');
        });
        kids.map(function (el, i) { return { el: el, i: i, r: rank(el.getAttribute('data-card')) }; })
            .sort(function (a, b) { return a.r - b.r || a.i - b.i; })
            .forEach(function (o) { row.appendChild(o.el); });
        const anyVisible = kids.some(function (el) { return el.style.display !== 'none'; });
        row.style.display = anyVisible ? '' : 'none';
    });
}

function cardSavePrefs() {
    const prefs = window.dashPrefs;
    try {
        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                cardVisible: prefs.visible || {},
                cardOrder: prefs.order || [],
                mode: prefs.mode || 'auto'
            })
        }).catch(function () {});
    } catch (e) {}
}

function loadCardPrefs() {
    return fetch('/api/settings').then(function (r) { return r.json(); }).then(function (s) {
        s = s || {};
        const prefs = window.dashPrefs;
        if (s.cardVisible && typeof s.cardVisible === 'object' && !Array.isArray(s.cardVisible)) {
            prefs.visible = s.cardVisible;
        }
        if (Array.isArray(s.cardOrder)) prefs.order = s.cardOrder;
        const m = s.mode || s.cardMode;
        prefs.mode = (m === 'onecol' || m === 'twocol') ? m : 'auto';
        applyCardLayout();
        return prefs;
    }).catch(function () { applyCardLayout(); });
}

function adminCardHost(slot) {
    const sec = document.getElementById('admin-sec-' + slot);
    if (!sec) return null;
    return sec.querySelector('.admin-slot') || sec;
}

// ── Admin section "Tarjetas": one checkbox per card ─────────────────────────
window.adminSections['cards'] = function () {
    const host = adminCardHost('cards');
    if (!host) return;
    const visible = window.dashPrefs.visible || {};
    const ids = cardOrderFull();
    let html = '<div class="ad-note">Tarjetas visibles</div>';
    ids.forEach(function (id) {
        html += '<label class="ad-check"><input type="checkbox" data-card-toggle="' + id + '"'
            + (visible[id] === false ? '' : ' checked') + '>'
            + '<span>' + (CARD_LABELS[id] || id) + '</span></label>';
    });
    host.innerHTML = html;
    host.querySelectorAll('[data-card-toggle]').forEach(function (cb) {
        cb.addEventListener('change', function () {
            const id = cb.getAttribute('data-card-toggle');
            window.dashPrefs.visible = window.dashPrefs.visible || {};
            window.dashPrefs.visible[id] = cb.checked;
            applyCardLayout();
            cardSavePrefs();
        });
    });
};

// ── Admin section "Diseño": column mode + card order + reset ────────────────
window.adminSections['layout'] = function () {
    const host = adminCardHost('layout');
    if (!host) return;
    const prefs = window.dashPrefs;
    const ids = cardOrderFull();
    const modes = [['auto', 'Auto'], ['onecol', '1 columna'], ['twocol', '2 columnas']];
    let html = '<div class="ad-note">Columnas por fila</div><div class="ad-modes">';
    modes.forEach(function (m) {
        html += '<label class="ad-radio"><input type="radio" name="ad-card-mode" value="' + m[0] + '"'
            + (prefs.mode === m[0] ? ' checked' : '') + '><span>' + m[1] + '</span></label>';
    });
    html += '</div><div class="ad-note">Orden de las tarjetas</div><div class="ad-order">';
    ids.forEach(function (id, i) {
        html += '<div class="ad-order-row"><span class="ad-order-name">' + (CARD_LABELS[id] || id) + '</span>'
            + '<span class="ad-order-btns">'
            + '<button type="button" class="ad-mini" data-card-up="' + id + '"' + (i === 0 ? ' disabled' : '') + ' title="Subir" aria-label="Subir">↑</button>'
            + '<button type="button" class="ad-mini" data-card-down="' + id + '"' + (i === ids.length - 1 ? ' disabled' : '') + ' title="Bajar" aria-label="Bajar">↓</button>'
            + '</span></div>';
    });
    html += '</div><div class="ad-actions"><button type="button" id="ad-card-reset">Restablecer</button></div>';
    host.innerHTML = html;

    host.querySelectorAll('input[name="ad-card-mode"]').forEach(function (rb) {
        rb.addEventListener('change', function () {
            if (!rb.checked) return;
            window.dashPrefs.mode = rb.value;
            applyCardLayout();
            cardSavePrefs();
        });
    });

    function move(id, delta) {
        const list = cardOrderFull();
        const i = list.indexOf(id);
        const j = i + delta;
        if (i < 0 || j < 0 || j >= list.length) return;
        list.splice(j, 0, list.splice(i, 1)[0]);
        window.dashPrefs.order = list;
        applyCardLayout();
        cardSavePrefs();
        adminRender('layout');
    }
    host.querySelectorAll('[data-card-up]').forEach(function (b) {
        b.addEventListener('click', function () { move(b.getAttribute('data-card-up'), -1); });
    });
    host.querySelectorAll('[data-card-down]').forEach(function (b) {
        b.addEventListener('click', function () { move(b.getAttribute('data-card-down'), 1); });
    });
    const resetBtn = host.querySelector('#ad-card-reset');
    if (resetBtn) resetBtn.addEventListener('click', function () {
        window.dashPrefs.visible = {};
        window.dashPrefs.order = CARD_NATURAL_ORDER.slice();
        window.dashPrefs.mode = 'auto';
        applyCardLayout();
        cardSavePrefs();
        adminRender('cards');
        adminRender('layout');
    });
};

// Hydrate from the backend right away (independent of the admin panel).
loadCardPrefs();

// ─── Modal helpers ─────────────────────────────────────────────────────────
function openModal(title, sub, html) {
    closeModal();
    const ov = document.createElement('div');
    ov.className = 'modal-overlay';
    ov.id = 'modal-overlay';
    ov.onclick = e => { if (e.target === ov) closeModal(); };
    ov.innerHTML = `
        <div class="modal">
            <h3>${title}</h3>
            ${sub ? `<div class="modal-sub">${sub}</div>` : ''}
            ${html}
            <button class="modal-close" onclick="closeModal()">Close</button>
        </div>`;
    document.body.appendChild(ov);
}
function closeModal() {
    const m = document.getElementById('modal-overlay');
    if (m) m.remove();
}

function openEnergyDetails() {
    const sp = lastMacmon.sys_power || 0;
    const gp = lastMacmon.gpu_power || 0;
    const meta = energyRangeMeta(energyRange);
    const hasAgg = !!(energyAgg && energyAgg.samples > 0);
    const kwh = hasAgg ? energyAgg.kwh : (energyRange === '1h' ? energyKWh() : 0);
    const cost = hasAgg ? energyAgg.cost : kwh * KWH_PRICE;
    const genTokens = lastSplash.decode_tokens || 0;
    const eff = (genTokens > 0 && kwh > 0) ? (kwh / genTokens * 1e6).toFixed(4) : '—';
    const elapsedH = (Date.now() - startTime) / 3600000;
    const min = elapsedH > 0.01 ? (elapsedH * 60).toFixed(0) + ' min' : 'just started';
    const rows = [
        ['Current power', `${sp.toFixed(1)} W`],
        ['GPU power', `${gp.toFixed(1)} W`],
        [`Energy · ${meta.unit}`, `${kwh.toFixed(3)} kWh`],
        ['Cost', `${cost.toFixed(3)} € @ ${KWH_PRICE.toFixed(2)} €/kWh`],
        ['Stored samples', hasAgg ? `${energyAgg.samples}` : '0 (session only)'],
        ['Efficiency', `${eff} kWh per 1k tokens`],
        ['Session duration', min],
    ];
    openModal('Energy details', 'macmon system power · trapezoid integration · stored history',
        `<table><tr><th>Metric</th><th>Value</th></tr>${rows.map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('')}</table>`);
}

function openTokenDetails() {
    const sp = lastSplash;
    const ol = lastOllama;
    const gen = sp.decode_tokens || 0, pre = sp.prefill_tokens || 0, ca = sp.cache_reused_tokens || 0;
    let rows = [
        ['Splash · Qwen3.8-27B-4bit', fmtToks(ca), fmtToks(pre), fmtToks(gen)],
    ];
    if (ol && ol.count > 0) {
        ol.models.forEach(m => rows.push([`Ollama · ${m.name}`, 'n/a', 'n/a', 'n/a']));
    }
    if (lastLlmstudio && lastLlmstudio.models) {
        lastLlmstudio.models.forEach(m => rows.push([`LLM Studio · ${m.id}`, 'n/a', 'n/a', `${fmtToks(m.gen || 0)} (${m.tps} t/s)`]));
    }
    openModal('Token details', 'Splash: from Prometheus counters · Ollama: no token API',
        `<table><tr><th>Model</th><th>Cached</th><th>Prefill</th><th>Generated</th></tr>` +
        rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join('') +
        `<tr class="total"><td>TOTAL (splash)</td><td>${fmtToks(ca)}</td><td>${fmtToks(pre)}</td><td>${fmtToks(gen)}</td></tr></table>`);
}

function openActivityDetails() {
    const items = [...activityLog].reverse();
    const html = items.length
        ? items.map(it => {
            const ago = Math.floor((Date.now() - it.t) / 60000);
            const t = ago < 1 ? 'now' : ago < 60 ? ago + 'm' : Math.floor(ago/60) + 'h ' + ago%60 + 'm';
            return `<li><span class="act-time">${t}</span><span class="act-dot ${it.color}"></span><span class="act-text">${it.text}</span></li>`;
        }).join('')
        : '<li><span class="act-text">No activity yet</span></li>';
    openModal('All activity', `${items.length} events since page load`, `<ul class="activity-list">${html}</ul>`);
}
</script>
</body>
</html>'''

MINIONS_HTML = '''<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Minions · live</title>
<style>
:root {
    --bg: #080a0f;
    --panel: #101520;
    --panel-2: #0c1019;
    --border: #212a38;
    --text: #e8eaed;
    --dim: #8f98a8;
    --mut: #4f5967;
    --green: #34a853;
    --orange: #f9ab00;
    --red: #ea4335;
    --blue: #4285f4;
    --purple: #a144fb;
    --teal: #14b8a6;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, "JetBrains Mono", Consolas, monospace;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { min-height: 100%; }
body {
    background-color: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-image:
        radial-gradient(1100px 520px at 12% -12%, rgba(52,168,83,0.10), transparent 62%),
        radial-gradient(900px 460px at 88% -6%, rgba(66,133,244,0.10), transparent 58%),
        radial-gradient(700px 500px at 50% 115%, rgba(161,68,251,0.07), transparent 60%),
        linear-gradient(rgba(255,255,255,0.020) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.020) 1px, transparent 1px);
    background-size: auto, auto, auto, 44px 44px, 44px 44px;
    background-attachment: fixed;
    min-height: 100vh;
}

/* ---------- header ---------- */
header {
    position: sticky; top: 0; z-index: 20;
    display: flex; align-items: center; justify-content: space-between;
    gap: 18px; flex-wrap: wrap;
    padding: 18px 30px 16px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(8,10,15,0.94), rgba(8,10,15,0.74));
    backdrop-filter: blur(10px);
}
.h-left { display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap; }
h1 {
    font-family: var(--mono);
    font-size: 21px; font-weight: 700; letter-spacing: 0.34em;
    color: #eaf6ee;
    text-shadow: 0 0 18px rgba(52,168,83,0.55), 0 0 44px rgba(52,168,83,0.22);
}
.h-sub { font-family: var(--mono); font-size: 11px; letter-spacing: 0.14em; color: var(--mut); text-transform: uppercase; }
.h-right { display: flex; align-items: center; gap: 14px; }
.live-wrap {
    display: inline-flex; align-items: center; gap: 9px;
    padding: 7px 13px 7px 11px;
    border: 1px solid rgba(52,168,83,0.34);
    border-radius: 999px;
    background: rgba(52,168,83,0.08);
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.22em; color: #9df3ba;
}
.live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 10px var(--green), 0 0 22px rgba(52,168,83,0.6);
    animation: beat 1.6s ease-in-out infinite;
}
.live-wrap.off { border-color: rgba(234,67,53,0.4); background: rgba(234,67,53,0.08); color: #ff9d93; }
.live-wrap.off .live-dot { background: var(--red); box-shadow: 0 0 10px var(--red); animation: none; }
.count {
    min-width: 40px; text-align: center;
    font-family: var(--mono); font-size: 13px; font-weight: 700;
    padding: 6px 11px; border-radius: 9px;
    color: #cdebd8; border: 1px solid rgba(52,168,83,0.28);
    background: rgba(52,168,83,0.09);
}
.count.zero { color: var(--mut); border-color: var(--border); background: rgba(255,255,255,0.03); }
.back {
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em;
    color: var(--dim); text-decoration: none;
    border: 1px solid var(--border); border-radius: 8px;
    padding: 7px 11px; background: rgba(255,255,255,0.03);
    transition: color .2s, border-color .2s, background .2s;
}
.back:hover { color: var(--text); border-color: rgba(52,168,83,0.4); background: rgba(52,168,83,0.08); }

@keyframes beat { 0%,100% { opacity:1; transform:scale(1);} 50% { opacity:.35; transform:scale(.82);} }

/* ---------- grid ---------- */
main { padding: 26px 30px 90px; max-width: 1680px; margin: 0 auto; }
#grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 18px;
}

/* ---------- card ---------- */
.mcard {
    --ac: var(--green);
    position: relative;
    border-radius: 15px;
    padding: 15px 17px 13px 19px;
    background: linear-gradient(158deg, rgba(19,24,34,0.96), rgba(11,14,21,0.98));
    border: 1px solid var(--border);
    box-shadow: 0 14px 34px rgba(0,0,0,0.42), inset 0 1px 0 rgba(255,255,255,0.03);
    overflow: hidden;
    transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.mcard::before {
    content: ""; position: absolute; left: 0; top: 12px; bottom: 12px; width: 2px;
    border-radius: 2px; background: var(--ac);
    box-shadow: 0 0 14px var(--ac), 0 0 32px var(--ac);
    opacity: .95;
}
.mcard::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(520px 120px at 0% 0%, color-mix(in srgb, var(--ac) 13%, transparent), transparent 70%);
    opacity: .55;
}
.mcard:hover { transform: translateY(-2px); border-color: color-mix(in srgb, var(--ac) 45%, var(--border)); box-shadow: 0 18px 44px rgba(0,0,0,0.5), 0 0 0 1px color-mix(in srgb, var(--ac) 22%, transparent); }
.mcard.enter { animation: cardIn .46s cubic-bezier(.18,.86,.24,1) both; }
.mcard.leave { animation: cardOut .3s ease-in forwards; pointer-events: none; }
@keyframes cardIn { from { opacity: 0; transform: translateY(14px) scale(.975); } to { opacity: 1; transform: none; } }
@keyframes cardOut { to { opacity: 0; transform: translateY(-8px) scale(.97); } }

.mcard[data-status="WORKING"]   { --ac: var(--green); }
.mcard[data-status="VERIFYING"] { --ac: var(--orange); }
.mcard[data-status="DONE"]      { --ac: var(--teal); }
.mcard[data-status="ERROR"]     { --ac: var(--red); }

.mtop { display: flex; align-items: center; gap: 10px; }
.chip {
    display: inline-flex; align-items: center; gap: 7px;
    font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: 0.16em;
    padding: 4px 9px 4px 8px; border-radius: 999px;
    color: var(--ac);
    border: 1px solid color-mix(in srgb, var(--ac) 38%, transparent);
    background: color-mix(in srgb, var(--ac) 11%, transparent);
}
.chip .d { width: 6px; height: 6px; border-radius: 50%; background: var(--ac); box-shadow: 0 0 8px var(--ac); }
.mcard[data-status="WORKING"] .chip .d { animation: beat 1.3s ease-in-out infinite; }
.mid {
    font-family: var(--mono); font-size: 10px; color: var(--mut);
    letter-spacing: 0.06em; margin-left: auto;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 45%;
}
.elapsed { font-family: var(--mono); font-size: 10.5px; color: var(--dim); letter-spacing: 0.08em; }

.goal {
    margin-top: 11px;
    font-size: 12.5px; line-height: 1.42; color: var(--dim);
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden;
}
.summary {
    margin-top: 9px;
    display: flex; align-items: baseline; gap: 8px;
    font-size: 16px; font-weight: 600; color: #eef4ff;
    letter-spacing: -0.01em;
    min-height: 22px;
}
.summary .caret { color: var(--ac); font-size: 12px; text-shadow: 0 0 10px var(--ac); transform: translateY(-1px); }
.sum-t { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.prow { display: flex; align-items: center; gap: 11px; margin-top: 14px; }
.pct {
    font-family: var(--mono); font-size: 13px; font-weight: 700; color: var(--ac);
    text-shadow: 0 0 14px color-mix(in srgb, var(--ac) 55%, transparent);
    min-width: 42px; letter-spacing: 0.02em;
}
.pbar {
    position: relative; flex: 1; height: 7px; border-radius: 999px;
    background: #1a2130; overflow: hidden;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.6);
}
.pbar-fill {
    position: relative; height: 100%; width: 0%; border-radius: 999px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--ac) 55%, #0a0d13), var(--ac));
    box-shadow: 0 0 12px var(--ac), 0 0 26px color-mix(in srgb, var(--ac) 45%, transparent);
    transition: width .7s cubic-bezier(.2,.8,.2,1);
    overflow: hidden;
}
.pbar-fill::after {
    content: ""; position: absolute; inset: 0;
    background: repeating-linear-gradient(115deg, rgba(255,255,255,0.22) 0 5px, transparent 5px 12px);
    animation: slide 1.1s linear infinite;
}
.mcard[data-status="DONE"] .pbar-fill::after,
.mcard[data-status="ERROR"] .pbar-fill::after { animation: none; opacity: .35; }
@keyframes slide { from { transform: translateX(-14px); } to { transform: translateX(0); } }
.stp { font-family: var(--mono); font-size: 10px; color: var(--mut); letter-spacing: 0.06em; white-space: nowrap; }

.mfoot {
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; margin-top: 13px; padding-top: 11px;
    border-top: 1px dashed rgba(255,255,255,0.07);
}
.toolchip {
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em;
    color: var(--dim);
    padding: 3px 7px; border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(255,255,255,0.035);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 55%;
}
.toolchip::before { content: "▸ "; color: var(--ac); }
.spark { width: 96px; height: 18px; flex: 0 0 auto; overflow: visible; }
.spark rect { fill: var(--ac); opacity: .28; transition: opacity .3s; }
.spark rect.hi { opacity: .95; filter: drop-shadow(0 0 3px var(--ac)); }
.spark-empty { font-family: var(--mono); font-size: 9px; color: var(--mut); letter-spacing: .1em; }

/* ---------- empty state ---------- */
#empty {
    display: none;
    flex-direction: column; align-items: center; justify-content: center;
    gap: 14px; padding: 110px 20px 130px; text-align: center;
}
#empty.show { display: flex; }
.empty-glyph {
    font-family: var(--mono); font-size: 58px; line-height: 1;
    color: #1d2431;
    text-shadow: 0 0 34px rgba(52,168,83,0.16);
    animation: drift 6s ease-in-out infinite;
}
@keyframes drift { 0%,100% { transform: translateY(0); opacity: .85; } 50% { transform: translateY(-7px); opacity: 1; } }
.empty-t { font-family: var(--mono); font-size: 13px; letter-spacing: 0.18em; color: var(--mut); text-transform: uppercase; }
.empty-s { font-family: var(--mono); font-size: 10.5px; color: #2f3846; letter-spacing: 0.1em; }

/* ---------- footer ---------- */
footer {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 15;
    display: flex; align-items: center; justify-content: space-between;
    gap: 12px; padding: 8px 30px;
    font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em;
    color: #3b4451;
    border-top: 1px solid rgba(255,255,255,0.05);
    background: linear-gradient(0deg, rgba(8,10,15,0.96), rgba(8,10,15,0.6));
    backdrop-filter: blur(8px);
}
footer .f-r { color: #2f3846; }
@media (max-width: 560px) {
    header { padding: 14px 16px; }
    main { padding: 18px 16px 90px; }
    #grid { grid-template-columns: 1fr; }
    h1 { font-size: 17px; letter-spacing: .24em; }
    footer { padding: 8px 16px; }
}
</style>
</head>
<body>
<header>
    <div class="h-left">
        <h1>MINIONS</h1>
        <span class="h-sub">hermes delegation · live</span>
    </div>
    <div class="h-right">
        <span class="live-wrap" id="live"><span class="live-dot"></span><span id="live-t">LIVE</span></span>
        <span class="count zero" id="count">0</span>
        <a class="back" href="/">← DASHBOARD</a>
    </div>
</header>

<main>
    <div id="grid"></div>
    <div id="empty">
        <div class="empty-glyph">◍</div>
        <div class="empty-t">No minions working — all quiet.</div>
        <div class="empty-s">waiting for the next delegation</div>
    </div>
</main>

<footer>
    <span id="src">src ~/.hermes/cache/delegation/live</span>
    <span class="f-r"><span id="tick">refresh 3s</span></span>
</footer>

<script>
const gridEl = document.getElementById('grid');
const emptyEl = document.getElementById('empty');
const countEl = document.getElementById('count');
const liveEl = document.getElementById('live');
const liveT = document.getElementById('live-t');
const tickEl = document.getElementById('tick');
const cards = new Map();
const refs = new Map();
let awaitingFirst = true;

function fmtElapsed(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    if (sec < 60) return sec + 's';
    const m = Math.floor(sec / 60), s = sec % 60;
    if (m < 60) return m + 'm ' + String(s).padStart(2, '0') + 's';
    const h = Math.floor(m / 60);
    return h + 'h ' + String(m % 60).padStart(2, '0') + 'm';
}

function sparkHTML(spark, ac) {
    const n = 12, W = 64, H = 16, gap = 1.6, bw = (W - gap * (n - 1)) / n;
    const max = Math.max(1, ...spark);
    let out = '';
    for (let i = 0; i < n; i++) {
        const v = spark[i] || 0;
        const h = v ? Math.max(2.2, (H - 2) * (v / max)) : 1.4;
        const x = i * (bw + gap);
        const hi = (v === max && v > 0 && i >= n - 4) ? ' class="hi"' : '';
        out += '<rect' + hi + ' x="' + x.toFixed(2) + '" y="' + (H - h).toFixed(2) +
               '" width="' + bw.toFixed(2) + '" height="' + h.toFixed(2) + '" rx="0.8"></rect>';
    }
    return out;
}

function makeCard(m) {
    const el = document.createElement('article');
    el.className = 'mcard enter';
    el.dataset.status = m.status;
    el.innerHTML =
        '<div class="mtop">' +
            '<span class="chip"><i class="d"></i><span class="chip-t"></span></span>' +
            '<span class="mid"></span>' +
            '<span class="elapsed"></span>' +
        '</div>' +
        '<div class="goal"></div>' +
        '<div class="summary"><span class="caret">▸</span><span class="sum-t"></span></div>' +
        '<div class="prow">' +
            '<span class="pct"></span>' +
            '<div class="pbar"><div class="pbar-fill"></div></div>' +
            '<span class="stp"></span>' +
        '</div>' +
        '<div class="mfoot"><span class="toolchip"></span>' +
            '<svg class="spark" viewBox="0 0 64 16" preserveAspectRatio="none"></svg>' +
        '</div>';
    const ref = {
        el: el,
        chipT: el.querySelector('.chip-t'),
        mid: el.querySelector('.mid'),
        elapsed: el.querySelector('.elapsed'),
        goal: el.querySelector('.goal'),
        sum: el.querySelector('.sum-t'),
        pct: el.querySelector('.pct'),
        fill: el.querySelector('.pbar-fill'),
        stp: el.querySelector('.stp'),
        tool: el.querySelector('.toolchip'),
        spark: el.querySelector('.spark')
    };
    refs.set(m.id, ref);
    gridEl.appendChild(el);
    setTimeout(() => el.classList.remove('enter'), 520);
    return ref;
}

function updateCard(ref, m) {
    const ac = m.status === 'ERROR' ? '#ea4335'
             : m.status === 'VERIFYING' ? '#f9ab00'
             : m.status === 'DONE' ? '#14b8a6' : '#34a853';
    ref.el.dataset.status = m.status;
    ref.chipT.textContent = m.status;
    ref.mid.textContent = m.id;
    ref.elapsed.textContent = m.status === 'DONE' ? 'took ' + fmtElapsed(m.elapsed) : fmtElapsed(m.elapsed);
    ref.goal.textContent = m.goal || '(no goal)';
    ref.goal.title = m.goal || '';
    ref.sum.textContent = m.summary || 'working';
    ref.pct.textContent = m.percent + '%';
    ref.fill.style.width = Math.max(2, Math.min(100, m.percent)) + '%';
    ref.stp.textContent = m.steps_total ? (m.tools + '/' + (m.steps_total * 3) + ' actions')
                                        : (m.steps + ' steps');
    ref.tool.textContent = m.last_action || 'think';
    ref.spark.innerHTML = sparkHTML(m.spark || new Array(12).fill(0), ac);
}

function render(list) {
    const seen = new Set();
    list.forEach((m, i) => {
        seen.add(m.id);
        let ref = refs.get(m.id);
        if (!ref) ref = makeCard(m);
        updateCard(ref, m);
        // keep newest-first order without tearing the DOM down
        if (gridEl.children[i] !== ref.el) {
            gridEl.insertBefore(ref.el, gridEl.children[i] || null);
        }
    });
    refs.forEach((ref, id) => {
        if (!seen.has(id)) {
            refs.delete(id);
            ref.el.classList.add('leave');
            setTimeout(() => ref.el.remove(), 320);
        }
    });
    countEl.textContent = list.length;
    countEl.classList.toggle('zero', list.length === 0);
    emptyEl.classList.toggle('show', list.length === 0 && !awaitingFirst);
    liveEl.classList.remove('off');
    liveT.textContent = 'LIVE';
}

function setOffline() {
    liveEl.classList.add('off');
    liveT.textContent = 'NO LINK';
}

async function poll() {
    if (document.hidden) return;
    try {
        const r = await fetch('/api/minions', { cache: 'no-store' });
        if (!r.ok) throw new Error('http ' + r.status);
        const list = await r.json();
        render(Array.isArray(list) ? list : []);
    } catch (e) {
        setOffline();
    } finally {
        awaitingFirst = false;
        emptyEl.classList.toggle('show', refs.size === 0 && !awaitingFirst);
        tickEl.textContent = 'updated ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }
}

poll();
setInterval(poll, 3000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });
</script>
</body>
</html>'''


def _page():
    """Return the dashboard HTML with the configured node/model/mem values injected.
    No-op when the env defaults are used, so the public repo stays generic and
    the service stays correct with a real `.env`."""
    html = DASHBOARD
    html = html.replace("Mac Mini M5 Pro", NODE_NAME)
    html = html.replace("54.5 GB", f"{MEMORY_TOTAL_GB} GB")
    html = html.replace("54.5e9", f"{MEMORY_TOTAL_GB}e9")
    html = html.replace("Qwen3.8-27B-4bit", MODEL_NAME)
    html = html.replace("192.168.1.x", NODE_IP)
    html = html.replace("__KWH_PRICE__", f"{KWH_PRICE}")
    return html

@app.route('/')
def dashboard():
    return _page()

@app.route('/api/splash')
def api_splash():
    return jsonify(metrics_cache.get('splash', {}))

@app.route('/api/macmon')
def api_macmon():
    return jsonify(macmon_cache.get('macmon', {}))

@app.route('/api/ollama')
def api_ollama():
    return jsonify(ollama_cache.get('ollama', {}))

@app.route('/api/llmstudio')
def api_llmstudio():
    return jsonify(llmstudio_cache.get('llmstudio', {}))

@app.route('/api/opencodego')
def api_opencodego():
    return jsonify(_go_cache or {})

# ============================================================================
# MINIONS — live Hermes subagent panel
# Reads the delegation live-transcript dir (plain-text or JSON transcripts) and
# exposes /api/minions + /minions. Nothing here touches the existing pollers.
# ============================================================================
DELEG_LIVE_DIR = os.getenv(
    "DASHSPLASH_DELEGATION_LIVE_DIR",
    "/home/ubuntu/.hermes/cache/delegation/live",
)
MINION_LIVE_WINDOW = float(os.getenv("DASHSPLASH_MINION_LIVE_WINDOW", "120"))

_MINION_LINE_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s+([a-z_]+)\s*\|\s?(.*)$")
_MINION_STEP_RE = re.compile(r"(?:^|[\s(])(\d{1,2})[).]\s+\S")

# Secrets that may leak from a child's transcript (config dumps, curl headers).
# Everything served by /api/minions goes through _minion_scrub().
_SECRET_PATTERNS = [
    (re.compile(r"oc_sk_[A-Za-z0-9_\-]+"), "[redacted]"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"), "[redacted]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[redacted]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[redacted]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[redacted]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"), "[redacted]"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "[redacted]"),
]
_SECRET_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{16,}")


def _minion_scrub(value):
    """Strip credentials from any string before it leaves the process."""
    if value is None:
        return ""
    s = str(value)
    for pat, repl in _SECRET_PATTERNS:
        s = pat.sub(repl, s)
    s = _SECRET_BEARER.sub(lambda m: m.group(1) + "[redacted]", s)
    return s


def _minion_role(t):
    return {
        "tool": "tool", "assistant": "assistant", "think": "think",
        "tool_result": "result", "result": "result", "error": "error",
        "summary": "assistant", "system": "system", "task": "start",
        "user": "user", "start": "start", "final": "final",
    }.get(t or "", t or "")


def _minion_read_log(path):
    """Parse one task-<N>.log. Supports the JSON transcript shape and the
    plain-text append-only shape. Returns None when unreadable/empty."""
    try:
        st = os.stat(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        return None
    body = raw.strip()
    if not body:
        return None

    parsed = None
    if body.startswith("{"):
        try:
            obj = json.loads(body)
            if isinstance(obj, dict) and isinstance(obj.get("transcript"), list):
                blocks = []
                for b in obj["transcript"]:
                    if not isinstance(b, dict):
                        continue
                    content = b.get("content")
                    if isinstance(content, (dict, list)):
                        content = json.dumps(content)[:400]
                    blocks.append({
                        "role": _minion_role(b.get("type")),
                        "text": str(content if content is not None else "")[:600],
                        "tool": b.get("tool"),
                        "ts": None,
                    })
                parsed = {
                    "goal": str(obj.get("goal") or ""),
                    "started": None,
                    "blocks": blocks,
                    "done": obj.get("result") is not None,
                    "error": None,
                }
        except (ValueError, TypeError):
            parsed = None

    if parsed is None:
        goal, started, blocks = "", None, []
        for line in body.splitlines():
            m = _MINION_LINE_RE.match(line)
            if m:
                hh, mm, ss, role, txt = m.groups()
                blocks.append({
                    "role": role,
                    "text": txt,
                    "tool": None,
                    "ts": int(hh) * 3600 + int(mm) * 60 + int(ss),
                })
                continue
            if not goal and line.startswith("goal:"):
                goal = line[5:].strip()
            elif not started and line.startswith("started:"):
                started = line[8:].strip()
        if not blocks and not goal:
            return None
        # done / error from the trailing "final" line, e.g. "final | end status=ok"
        tail = " ".join(b["text"] for b in blocks[-3:] if b["role"] in ("final", "error"))
        done = bool(re.search(r"status=(ok|completed|done|success)", tail, re.I))
        error = None
        if blocks and blocks[-1]["role"] == "error":
            error = blocks[-1]["text"]
        m = re.search(r"status=([a-z_]+)", tail, re.I)
        if m and m.group(1).lower() in ("failed", "error", "interrupted", "cancelled"):
            error = error or f"status={m.group(1)}"
        parsed = {
            "goal": goal, "started": started, "blocks": blocks,
            "done": done or (bool(blocks) and blocks[-1]["role"] == "final" and not error),
            "error": error,
        }

    parsed["mtime"] = st.st_mtime
    parsed["size"] = st.st_size
    # normalise timestamps to epoch seconds (seconds-of-day -> today, with a
    # midnight guard so a log started at 23:58 still reads correctly at 00:05).
    now = time.time()
    today0 = now - (now % 86400) + time.timezone  # local midnight, cheap approx
    for b in parsed["blocks"]:
        if b["ts"] is not None and b["ts"] < 86400:
            b["ts"] = today0 + b["ts"]
            if b["ts"] > now + 120:
                b["ts"] -= 86400
    return parsed


_MINION_CACHE = {}


def _minion_log(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size)
    hit = _MINION_CACHE.get(path)
    if hit and hit[0] == key:
        return hit[1]
    data = _minion_read_log(path)
    _MINION_CACHE[path] = (key, data)
    return data


_TERMINAL_VERBS = [
    (r"git push", "Pushing to git"),
    (r"git commit", "Committing changes"),
    (r"git add", "Staging files"),
    (r"git (status|diff|log|show)", "Checking git state"),
    (r"localhost:8081|127\.0\.0\.1:8081|/api/", "Probing the dashboard API"),
    (r"curl", "Probing HTTP endpoints"),
    (r"systemctl|service ", "Restarting the service"),
    (r"pytest|unittest|npm test|go test", "Running the test suite"),
    (r"\bpip install|\bbrew install|apt-get", "Installing dependencies"),
    (r"\bpython[0-9.]*\b", "Running a Python script"),
    (r"\b(sed|awk)\b", "Reading source slices"),
    (r"\b(grep|rg)\b", "Grepping the codebase"),
    (r"\b(find|ls)\b", "Listing files"),
    (r"\bcat\b", "Reading a file"),
    (r"\b(wc|du|df)\b", "Measuring files"),
    (r"\bmkdir|\bcp\b|\bmv\b", "Moving files around"),
    (r"\bssh|scp|rsync", "Reaching another host"),
    (r"\bdocker|kubectl", "Touching containers"),
]
_TOOL_LABELS = {
    "read_file": "Reading", "write_file": "Writing", "patch": "Editing",
    "search_files": "Searching code", "web_search": "Searching the web",
    "web_extract": "Fetching a page", "skill_view": "Loading a skill",
    "todo_list": "Updating the plan", "delegate_task": "Spawning a child",
    "execute_code": "Running Python", "browser_exec": "Driving the browser",
    "vision_analyze": "Looking at an image", "terminal": "Running",
    "process_manage": "Checking a process", "skill_manage": "Saving a skill",
}


def _minion_from_tool(text, tool=None):
    """Turn '-> terminal(git status --short)' into 'Checking git state'."""
    raw = (text or "").strip()
    arg = ""
    name = (tool or "").strip()
    m = re.match(r"^->\s*([a-z_]+)\((.*)\)\s*$", raw, re.S)
    if m:
        name = m.group(1) or name
        arg = m.group(2)
    elif raw.startswith("->"):
        arg = raw[2:].strip()
    arg = arg.strip()
    if name == "terminal" or name == "":
        low = raw.lower()
        for pat, label in _TERMINAL_VERBS:
            if re.search(pat, low):
                return label
        first = arg.split()[0] if arg.split() else "command"
        if first.strip("'\"") in ("echo", "printf", "cd", "true", "false", "sleep",
                                   "source", "export", "set", "test", "[", "fi",
                                   "done", "if", "for", "while", "then", "do"):
            return "Running a shell command"
        return f"Running {first}"[:48]
    label = _TOOL_LABELS.get(name)
    if not label:
        return name.replace("_", " ").capitalize()
    if name in ("read_file", "write_file", "patch", "seek") and arg:
        base = arg.split("(")[0].replace("'", " ").replace('"', " ").strip()
        base = os.path.basename(base.split()[0]) if base.split() else ""
        if base:
            return f"{label} {base}"[:48]
    if name in ("search_files", "web_search") and arg:
        return f"{label}: {arg.split(',')[0][:34]}"
    return label


def _minion_words(text, limit=9):
    """First `limit` words of a prose block, markdown stripped."""
    s = re.sub(r"[`*_#>\[\]]", "", text or "")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    words = s.split(" ")
    out = " ".join(words[:limit])
    if len(words) > limit:
        out += "…"
    return out[:110]


_VERIFY_RE = re.compile(
    r"curl\s+-s|curl\s+-I|systemctl status|journalctl|pytest|\bjest\b|\bverify|"
    r"\bassert\b|diff\s+-|<\(|grep -c|json\.load|json\.tool",
    re.I,
)


def _minion_goal_steps(goal):
    nums = {int(n) for n in _MINION_STEP_RE.findall(goal or "") if 1 <= int(n) <= 20}
    return max(nums) if nums else 0


def _minion_build(deleg_id, task_idx, data, manifest):
    blocks = data["blocks"] or []
    now = time.time()

    # --- current action -----------------------------------------------------
    last_tool = None
    last_tool_i = -1
    for i, b in enumerate(blocks):
        if b["role"] == "tool":
            last_tool, last_tool_i = b, i
    summary, tool_name = "", ""
    if last_tool is not None:
        summary = _minion_from_tool(last_tool["text"], last_tool.get("tool"))
        tool_name = re.sub(r"^->\s*([a-z_]+)\(.*", r"\1", last_tool["text"].strip()) or "tool"
    generic = summary in ("Running a shell command", "Running terminal command", "Running sh", "")
    if not summary or generic:
        # a truncated shell blob says nothing useful -> fall back to the last
        # sentence the agent actually wrote about what it is doing
        for b in list(reversed(blocks))[-8:]:
            if b["role"] in ("assistant", "think"):
                prose = _minion_words(b["text"], 8)
                if len(prose) > 12:
                    summary = prose
                    break
    if not summary:
        summary = "Warming up"
    if not tool_name:
        tool_name = "think"

    # --- status -------------------------------------------------------------
    manifest_status = (manifest.get("status") or "").lower()
    last_role = blocks[-1]["role"] if blocks else ""
    errored = bool(data.get("error")) or manifest_status in ("failed", "error", "interrupted", "cancelled")
    if errored:
        status = "ERROR"
    elif data.get("done") or manifest_status in ("completed", "done", "success"):
        status = "DONE"
    elif last_tool is not None and last_tool_i == len(blocks) - 1 and _VERIFY_RE.search(last_tool["text"] or ""):
        status = "VERIFYING"
    elif last_role in ("result",) and last_tool is not None and _VERIFY_RE.search(last_tool["text"] or ""):
        status = "VERIFYING"
    else:
        status = "WORKING"

    # --- progress -----------------------------------------------------------
    tools_done = sum(1 for b in blocks if b["role"] == "tool")
    total_steps = _minion_goal_steps(data.get("goal") or "")
    if status == "DONE":
        percent = 100
    elif total_steps >= 2:
        done_steps = min(total_steps, tools_done / 3.0)
        percent = int(max(3, min(95, 100.0 * done_steps / total_steps)))
    else:
        percent = int(max(4, min(95, tools_done * 12)))
    if status == "ERROR":
        percent = max(percent, 5)

    # --- timing -------------------------------------------------------------
    stamps = [b["ts"] for b in blocks if b.get("ts")]
    started_epoch = stamps[0] if stamps else data["mtime"]
    started = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_epoch))
    elapsed = max(0, int((now if status in ("WORKING", "VERIFYING") else data["mtime"]) - started_epoch))

    # --- sparkline: steps per 10s bucket over the last 2 minutes -------------
    spark = [0] * 12
    for b in blocks:
        ts = b.get("ts")
        if not ts or b["role"] in ("user", "start", "system"):
            continue
        age = now - ts
        if 0 <= age < 120:
            spark[11 - int(age // 10)] += 1

    return {
        "id": f"{deleg_id}#{task_idx}",
        "goal": _minion_scrub((data.get("goal") or "")[:400]),
        "summary": _minion_scrub(summary),
        "percent": percent,
        "status": status,
        "started": started,
        "elapsed": elapsed,
        "last_action": _minion_scrub(tool_name),
        "steps": len(blocks),
        "tools": tools_done,
        "spark": spark,
        "steps_total": total_steps,
    }


def scan_minions():
    """Every live delegation task, newest activity first."""
    out = []
    try:
        entries = os.listdir(DELEG_LIVE_DIR)
    except OSError:
        return out
    now = time.time()
    for deleg_id in entries:
        ddir = os.path.join(DELEG_LIVE_DIR, deleg_id)
        if not os.path.isdir(ddir):
            continue
        manifest = {}
        try:
            with open(os.path.join(ddir, "manifest.json"), "r", encoding="utf-8") as fh:
                manifest = json.load(fh) or {}
        except (OSError, ValueError):
            manifest = {}
        tasks_meta = {t.get("index"): t for t in (manifest.get("tasks") or []) if isinstance(t, dict)}
        try:
            names = os.listdir(ddir)
        except OSError:
            continue
        for name in names:
            m = re.match(r"^task-(\d+)\.log$", name)
            if not m:
                continue
            idx = int(m.group(1))
            path = os.path.join(ddir, name)
            try:
                if now - os.stat(path).st_mtime > MINION_LIVE_WINDOW:
                    continue
            except OSError:
                continue
            data = _minion_log(path)
            if not data:
                continue
            meta = dict(manifest)
            meta.update({k: v for k, v in (tasks_meta.get(idx) or {}).items() if v})
            item = _minion_build(deleg_id, idx, data, meta)
            item["mtime"] = int(data["mtime"])
            out.append(item)
    out.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return out


@app.route("/api/minions")
def api_minions():
    try:
        return jsonify(scan_minions())
    except Exception as e:  # never 500 the panel
        print(f"[MINIONS] scan failed: {e}", flush=True)
        return jsonify([])


@app.route("/minions")
def minions_page():
    return MINIONS_HTML

@app.route('/api/splash_status')
def api_splash_status():
    return jsonify(splash_status_cache.get('splash_status', {}))

@app.route('/api/release', methods=['POST'])
def api_release():
    """Release a model from RAM on demand.
    body: {"engine": "splash"}  -> POST /v1/memory/release (splash 1.3.2+)
          {"engine": "ollama"}  -> POST /api/chat keep_alive:0 for each loaded model
    """
    engine = (request.get_json(silent=True) or {}).get('engine', '')
    try:
        if engine == 'splash':
            r = requests.post(f"{SPLASH_BASE}/v1/memory/release", timeout=10)
            return jsonify({"engine": "splash", "ok": r.status_code in (200, 202), "status": r.status_code, "response": r.json() if r.headers.get('content-type','').startswith('application/json') else r.text[:200]})
        if engine == 'ollama':
            ol = ollama_cache.get('ollama', {})
            unloaded = []
            for mo in ol.get('models', []):
                try:
                    cr = requests.post(f"{OLLAMA_BASE}/api/chat", json={
                        "model": mo['name'],
                        "messages": [{"role": "user", "content": "bye"}],
                        "stream": False,
                        "keep_alive": 0,
                    }, timeout=60)
                    unloaded.append({"model": mo['name'], "ok": cr.status_code == 200})
                except Exception as e:
                    unloaded.append({"model": mo['name'], "ok": False, "error": str(e)[:120]})
            return jsonify({"engine": "ollama", "ok": True, "models": unloaded})
        if engine == 'llmstudio':
            ls = llmstudio_cache.get('llmstudio', {})
            base = LLMSTUDIO_URL.rstrip('/')
            unloaded = []
            for mo in ls.get('models', []):
                try:
                    cr = requests.post(f"{base}/api/v0/models/{mo['id']}/unload", timeout=15)
                    unloaded.append({"model": mo['id'], "ok": cr.status_code in (200, 202)})
                except Exception as e:
                    unloaded.append({"model": mo['id'], "ok": False, "error": str(e)[:120]})
            return jsonify({"engine": "llmstudio", "ok": True, "models": unloaded})
        return jsonify({"error": "engine must be 'splash', 'ollama' or 'llmstudio'"}), 400
    except Exception as e:
        return jsonify({"engine": engine, "ok": False, "error": str(e)[:200]}), 502

# ─── Settings store (theme, refresh, background, card order/visibility, …) ──
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DASHSPLASH_DATA_DIR", os.path.join(BASE_DIR, "data"))
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "theme": "dark",
    "refreshMs": 5000,
    "bgFile": None,
    "cardOrder": [],
    "cardVisible": {},
    "energyRange": "1h",
}

SETTINGS = dict(DEFAULT_SETTINGS)


def load_settings():
    """Load settings.json into the module-level SETTINGS dict (defaults first)."""
    global SETTINGS
    merged = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in DEFAULT_SETTINGS.items()}
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    merged[k] = v          # keep known keys AND extra keys added elsewhere
    except Exception as e:
        print(f"[SETTINGS] load failed ({SETTINGS_FILE}): {e}", flush=True)
    SETTINGS = merged
    return SETTINGS


def save_settings(patch=None):
    """Merge `patch` into SETTINGS and overwrite settings.json atomically."""
    global SETTINGS
    if isinstance(patch, dict):
        SETTINGS.update(patch)
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = SETTINGS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(SETTINGS, f, indent=2, ensure_ascii=False)
        os.replace(tmp, SETTINGS_FILE)     # overwrite
        return True
    except Exception as e:
        print(f"[SETTINGS] save failed ({SETTINGS_FILE}): {e}", flush=True)
        return False


SETTINGS = load_settings()
print(f"[START] settings loaded from {SETTINGS_FILE}: theme={SETTINGS.get('theme')} "
      f"refreshMs={SETTINGS.get('refreshMs')}", flush=True)


@app.route('/api/settings', methods=['GET'])
def api_settings_get():
    return jsonify(SETTINGS)


@app.route('/api/settings', methods=['POST'])
def api_settings_post():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "body must be a JSON object"}), 400
    save_settings(body)
    return jsonify(SETTINGS)


# ─── Custom background image: upload / query / serve ─────────────────────────
BG_DIR = os.path.join(DATA_DIR, "bg")
ALLOWED_BG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}
MAX_BG_BYTES = 25 * 1024 * 1024


def _bg_safe_name(name):
    """Return a safe basename or None (no path traversal)."""
    if not name:
        return None
    name = str(name)
    base = os.path.basename(name)
    if base != name or base in (".", "..") or base.startswith("."):
        return None
    if os.path.splitext(base)[1].lower() not in ALLOWED_BG_EXT:
        return None
    return base


def _bg_current_filename():
    """Filename stored in SETTINGS.bgFile that actually exists on disk."""
    name = _bg_safe_name(SETTINGS.get("bgFile"))
    if not name:
        return None
    return name if os.path.isfile(os.path.join(BG_DIR, name)) else None


@app.route('/api/bg', methods=['GET'])
def api_bg_get():
    return jsonify({"ok": True, "filename": _bg_current_filename()})


@app.route('/api/bg', methods=['POST'])
def api_bg_post():
    """Accept a multipart file field 'image' OR JSON {image: 'data:image/...;base64,...'}.
    JSON {clear: true} removes the background. Persists SETTINGS.bgFile."""
    import base64
    from flask import send_from_directory  # noqa: F401  (kept local: used by /bg route)

    os.makedirs(BG_DIR, exist_ok=True)
    raw = None
    ext = None

    up = request.files.get('image')
    if up is not None and up.filename:
        raw = up.read()
        ext = os.path.splitext(up.filename)[1].lower()
        if ext not in ALLOWED_BG_EXT:
            mt = (up.mimetype or '').lower()
            ext = {'.png': '.png', '.jpeg': '.jpg', '.webp': '.webp',
                   '.gif': '.gif', '.bmp': '.bmp', '.avif': '.avif'}.get(
                       '.' + mt.split('/')[-1], None)
    else:
        body = request.get_json(silent=True)
        if isinstance(body, dict) and body.get('clear'):
            save_settings({"bgFile": None})
            return jsonify({"ok": True, "filename": None})
        data = None
        if isinstance(body, dict):
            data = body.get('image') or body.get('dataUrl') or body.get('data')
            name = body.get('name') or 'bg'
            ext = os.path.splitext(str(name))[1].lower() or None
        if isinstance(data, str):
            if data.startswith('data:'):
                header, _, payload = data.partition(',')
                mime = header[5:].split(';')[0].strip().lower()
                if not ext:
                    ext = '.' + mime.split('/')[-1] if mime.startswith('image/') else None
                try:
                    raw = base64.b64decode(payload, validate=False)
                except Exception:
                    raw = None
            else:                                  # bare base64
                try:
                    raw = base64.b64decode(data, validate=False)
                except Exception:
                    raw = None

    if not raw:
        return jsonify({"ok": False, "error": "no image received (field 'image' or JSON data URL)"}), 400
    if len(raw) > MAX_BG_BYTES:
        return jsonify({"ok": False, "error": "image too large (max 25 MB)"}), 413

    # Normalise extension by magic bytes when the client gave us nothing usable.
    if ext not in ALLOWED_BG_EXT:
        if raw[:3] == b'\xff\xd8\xff':
            ext = '.jpg'
        elif raw[:8] == b'\x89PNG\r\n\x1a\n':
            ext = '.png'
        elif raw[:4] == b'RIFF' and raw[8:12] == b'WEBP':
            ext = '.webp'
        elif raw[:6] in (b'GIF87a', b'GIF89a'):
            ext = '.gif'
        else:
            ext = '.jpg'
    if ext == '.jpeg':
        ext = '.jpg'

    filename = 'bg' + ext
    # Drop previous background files so data/bg never accumulates.
    try:
        for old in os.listdir(BG_DIR):
            if old.lower().startswith('bg.'):
                try:
                    os.remove(os.path.join(BG_DIR, old))
                except OSError:
                    pass
    except OSError:
        pass

    path = os.path.join(BG_DIR, filename)
    try:
        with open(path, 'wb') as f:
            f.write(raw)
    except Exception as e:
        return jsonify({"ok": False, "error": f"write failed: {e}"}), 500

    save_settings({"bgFile": filename})
    print(f"[BG] stored {filename} ({len(raw)} bytes) -> SETTINGS.bgFile", flush=True)
    return jsonify({"ok": True, "filename": filename, "size": len(raw)})


@app.route('/bg/<path:filename>', methods=['GET'])
def serve_bg(filename):
    from flask import send_from_directory
    name = _bg_safe_name(filename)
    if not name:
        return jsonify({"error": "not found"}), 404
    if not os.path.isfile(os.path.join(BG_DIR, name)):
        return jsonify({"error": "not found"}), 404
    resp = send_from_directory(BG_DIR, name, max_age=0)
    resp.headers['Cache-Control'] = 'no-cache, must-revalidate'
    return resp


# ─── Energy history API (power samples persisted under data/energy/) ────────
@app.route('/api/energy/history')
def api_energy_history():
    """Downsampled power points for a range (default: settings energyRange).

    ?range=1h|24h|7d|30d|90d  →  {"range","start","end","samples","kwh","cost",
                                    "points":[{t,watts,gpu_watts}, ...]}
    """
    rng = str(request.args.get('range') or SETTINGS.get('energyRange') or '1h').strip()
    if rng not in ENERGY_RANGES:
        return jsonify({"error": "range must be one of " + ", ".join(ENERGY_RANGES)}), 400
    return jsonify(_energy_query(rng))


@app.route('/api/energy/summary')
def api_energy_summary():
    """Aggregate kWh + cost for a range (no points payload)."""
    rng = str(request.args.get('range') or SETTINGS.get('energyRange') or '1h').strip()
    if rng not in ENERGY_RANGES:
        return jsonify({"error": "range must be one of " + ", ".join(ENERGY_RANGES)}), 400
    q = _energy_query(rng) or {}
    return jsonify({
        "range": rng,
        "start": q.get("start"),
        "end": q.get("end"),
        "samples": q.get("samples", 0),
        "kwh": q.get("kwh", 0.0),
        "cost": q.get("cost", 0.0),
        "kwhPrice": KWH_PRICE,
    })


@app.route('/api/energy/clear', methods=['POST'])
def api_energy_clear():
    """Wipe the persisted power history (used by the admin 'borrar historial')."""
    with _ENERGY_LOCK:
        rows = _energy_load()
        removed = len(rows)
        rows.clear()
        _energy_rewrite(rows)
    print(f"[ENERGY] history cleared ({removed} samples)", flush=True)
    return jsonify({"ok": True, "removed": removed})


if __name__ == '__main__':
    print(f"{NODE_NAME} Monitor on 0.0.0.0:{PORT}")
    # Load persisted settings (theme/bg/cards/refresh) before serving.
    load_settings()
    # Seed the LLM Studio cache synchronously so the very first request
    # (and the first dashboard render) shows real data instead of a FAIL.
    try:
        _ls = fetch_llmstudio()
        if _ls:
            llmstudio_cache['llmstudio'] = _ls
            print(f"[START] LLM Studio seeded: {_ls.get('count')} models, {_ls.get('tps')} tps")
    except Exception as e:
        print(f"[START] LLM Studio seed failed: {e}")
    # Seed the OpenCode Go quota cache too (never raises).
    try:
        _go = fetch_opencodego()
        print(f"[START] OpenCode Go seeded: online={_go.get('online')} reason={_go.get('reason')}")
    except Exception as e:
        print(f"[START] OpenCode Go seed failed: {e}")
    app.run(host='0.0.0.0', port=PORT, threaded=True)
