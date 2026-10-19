import os
import time
import requests
import threading
import re
import sys
from flask import Flask, render_template_string, jsonify, request
from urllib.parse import urlparse

app = Flask(__name__)

# ---- Config (override via env vars, or a `.env` file sourced by the launcher) ----
# Defaults are generic so the repo is safe to publish; put your real values in `.env`.
SPLASH_URL        = os.getenv("DASHSPLASH_SPLASH_URL", "http://127.0.0.1:8000/metrics")
MACMON_URL        = os.getenv("DASHSPLASH_MACMON_URL", "http://127.0.0.1:9090/json")
OLLAMA_URL        = os.getenv("DASHSPLASH_OLLAMA_URL", "http://127.0.0.1:11434/api/ps")
MODEL_NAME        = os.getenv("DASHSPLASH_MODEL",      "Qwen3.8-27B-4bit")
GPU_TDP           = float(os.getenv("DASHSPLASH_GPU_TDP",  "100"))    # GPU TDP in W
MEMORY_TOTAL_GB   = float(os.getenv("DASHSPLASH_MEM_GB",  "54.5"))    # unified memory in GB
NODE_NAME         = os.getenv("DASHSPLASH_NODE_NAME",  "Mac Mini M5 Pro")
NODE_IP           = os.getenv("DASHSPLASH_NODE_IP",    "192.168.1.x")
PORT              = int(os.getenv("DASHSPLASH_PORT",   "8081"))

# Base URLs (host:port only) so we can hit control endpoints that live
# beside the metrics/ps paths. Used by the "release model" buttons.
def _base(url):
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"
SPLASH_BASE = _base(SPLASH_URL)   # http://192.168.1.x:8000
OLLAMA_BASE = _base(OLLAMA_URL)   # http://192.168.1.x:11434

metrics_cache = {}
macmon_cache = {}
ollama_cache = {}
history = {
    'memory_used': [], 'memory_limit': [],
    'decode_tps': [], 'prefill_tps': [],
    'ttft_p50': [], 'itl_p50': [],
    'gpu_temp': [], 'cpu_temp': [],
    'gpu_power': [], 'sys_power': [],
    'gpu_util': [], 'gpu_freq': [],
    'fan_rpm': [],
}

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

def update_loop():
    global metrics_cache, macmon_cache, ollama_cache
    while True:
        try:
            splash = fetch_data()
            macmon = fetch_macmon()
            ollama = fetch_ollama()
            record_history(splash, macmon)
            metrics_cache['splash'] = splash
            macmon_cache['macmon'] = macmon
            ollama_cache['ollama'] = ollama
        except:
            pass
        time.sleep(5)

threading.Thread(target=update_loop, daemon=True).start()

DASHBOARD = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mac Monitor</title>
<style>
:root {
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
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    padding: 20px 24px; min-height: 100vh;
}

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
.node-badge.worker { background: rgba(255,255,255,0.06); color: var(--text-dim); }
.node-ip { margin-left: auto; font-size: 11px; color: var(--text-mut); }

/* Memory section */
.mem-section { margin-bottom: 12px; }
.mem-label { font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
.mem-free { font-size: 14px; font-weight: 700; margin-bottom: 6px; }
.mem-bar { height: 8px; background: #2a2f3a; border-radius: 4px; overflow: hidden; display: flex; }
.mem-bar-splash { background: var(--green); height: 100%; }
.mem-bar-ollama { background: var(--blue); height: 100%; }
.mem-bar-sys { background: var(--purple); height: 100%; }
.mem-bar-free { background: #3a3f4a; height: 100%; }
.mem-legend { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 5px; font-size: 10px; color: var(--text-dim); }
.mem-legend span::before { content: '● '; }
.mem-legend .lg-splash::before { color: var(--green); }
.mem-legend .lg-ollama::before { color: var(--blue); }
.mem-legend .lg-sys::before { color: var(--purple); }
.mem-legend .lg-free::before { color: #5a6070; }

/* GPU metrics row */
.gpu-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 14px; }
.gpu-item { }
.gpu-label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.gpu-val { font-size: 16px; font-weight: 700; margin-bottom: 5px; }
.gpu-val .unit { font-size: 11px; font-weight: 400; color: var(--text-dim); }
.bar { height: 6px; background: #2a2f3a; border-radius: 3px; overflow: hidden; }
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
.model-name { font-size: 13px; font-weight: 600; color: #e8d5a3; }
.model-ctx { font-size: 10px; color: var(--text-dim); background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 4px; }
.release-btn {
    margin-left: 8px; cursor: pointer; user-select: none;
    font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
    color: #ff8a8a; background: rgba(255,120,120,0.12);
    border: 1px solid rgba(255,120,120,0.35); border-radius: 6px;
    padding: 4px 9px; transition: all 0.15s;
}
.release-btn:hover { background: rgba(255,120,120,0.28); }
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
.stat-bar { height: 5px; background: #2a2f3a; border-radius: 3px; margin-top: 8px; overflow: hidden; }
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
.bc-badge { font-size: 10px; color: var(--text-dim); background: rgba(255,255,255,0.06); padding: 2px 6px; border-radius: 4px; }
.bc-btn { font-size: 10px; color: var(--text-dim); background: rgba(255,255,255,0.08); border: none; padding: 3px 8px; border-radius: 4px; cursor: pointer; margin-left: auto; }
.bc-btn:hover { background: rgba(255,255,255,0.14); }
.bc-big { font-size: 26px; font-weight: 700; }
.bc-big .unit { font-size: 12px; font-weight: 400; color: var(--text-dim); }
.bc-sub { font-size: 10px; color: var(--text-mut); margin: 4px 0 10px; }
.bc-chart { height: 80px; margin-bottom: 10px; }
.bc-chart canvas { width: 100%; height: 80px; }
.bc-footer { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
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
.chart-title { font-size: 11px; color: var(--text-dim); margin-bottom: 8px; display: flex; justify-content: space-between; }
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
    <div class="header-right" id="header-status"></div>
</div>

<!-- NODE CARD -->
<div id="node-card"></div>

<!-- AGGREGATE STATS -->
<div class="stats-row" id="stats-row"></div>

<!-- BOTTOM ROW -->
<div class="bottom-row" id="bottom-row"></div>

<!-- CHARTS -->
<div class="charts-row" id="charts-row"></div>

<script>
const REFRESH_MS = 5000;
const history = {};
const historyKeys = ['memory_used','memory_limit','decode_tps','prefill_tps','ttft_p50','itl_p50','gpu_temp','cpu_temp','gpu_power','sys_power','gpu_util','gpu_freq','fan_rpm'];
historyKeys.forEach(k => history[k] = []);
let lastSplash = {}, lastMacmon = {}, lastOllama = {};
let activityLog = [];
let startTime = Date.now();
let lastEnergy = 0;

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

function renderNode(splash, macmon, ollama) {
    const card = document.getElementById('node-card');
    ollama = ollama || {};
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
    const totGB = gb(rmTot);
    const freeGB = Math.max(0, totGB - gb(rmUsed));
    const splashGB = gb(mu);
    // segmentos de la barra (acotados a 100%)
    let sPct = totGB > 0 ? Math.min(100, splashGB / totGB * 100) : 0;
    let oPct = totGB > 0 ? Math.min(Math.max(0, 100 - sPct), ollGB / totGB * 100) : 0;
    let sysGB = Math.max(0, gb(rmUsed) - splashGB - ollGB);
    let sysPct = totGB > 0 ? Math.min(Math.max(0, 100 - sPct - oPct), sysGB / totGB * 100) : 0;
    let freePct = Math.max(0, 100 - sPct - oPct - sysPct);
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
                <div class="mem-bar-sys" style="width:${sysPct}%"></div>
                <div class="mem-bar-free" style="width:${freePct}%"></div>
            </div>
            <div class="mem-legend">
                <span class="lg-splash">Splash ${splashGB.toFixed(1)} GB</span>
                <span class="lg-ollama">Ollama ${ollGB.toFixed(1)} GB</span>
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
    <div class="stat-card">
        <div class="stat-label">MAC DECODE</div>
        <div class="stat-value">${dtps.toFixed(1)} <span class="unit">tok/s</span></div>
        <div class="stat-spark"><canvas id="spark-decode"></canvas></div>
    </div>
    <div class="stat-card">
        <div class="stat-label">MAC POWER</div>
        <div class="stat-value">${sp.toFixed(0)} <span class="unit">W</span></div>
        <div class="stat-spark"><canvas id="spark-power"></canvas></div>
    </div>
    <div class="stat-card">
        <div class="stat-label">MEMORY IN USE</div>
        <div style="display:flex;align-items:baseline;gap:6px">
            <div class="stat-value" style="color:var(--blue)">${memPct.toFixed(0)}%</div>
            <div style="font-size:13px;color:var(--text-dim)">${gb(mu).toFixed(1)} GB</div>
        </div>
        <div class="stat-bar"><div class="stat-bar-fill" style="width:${memPct}%;background:var(--blue)"></div></div>
    </div>
    <div class="stat-card">
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

function renderBottom(splash, macmon) {
    const row = document.getElementById('bottom-row');
    const sp = macmon.sys_power || 0;
    const gp = macmon.gpu_power || 0;
    const dtps = splash.decode_tps || 0;
    const genTokens = splash.decode_tokens || 0;
    const prefillTokens = splash.prefill_tokens || 0;
    const cachedTokens = splash.cache_reused_tokens || 0;
    const totalTokens = genTokens + cachedTokens + prefillTokens;

    // Energy estimate (kWh since dashboard started)
    const elapsedH = (Date.now() - startTime) / 3600000;
    const energyKWh = elapsedH > 0 ? (sp * elapsedH / 1000) : 0;
    const efficiency = genTokens > 0 ? (energyKWh / genTokens * 1e6).toFixed(4) : '—';

    let activityHTML;
    const actItems = activityLog.slice(-8).reverse();
    if (actItems.length === 0) {
        activityHTML = '<li><span class="act-time">—</span><span class="act-dot blue"></span><span class="act-text">Monitoring started</span></li>';
    } else {
        activityHTML = actItems.map(item => {
            const ago = Math.floor((Date.now() - item.t) / 60000);
            const timeStr = ago < 1 ? 'now' : ago < 60 ? ago + 'm' : Math.floor(ago/60) + 'h';
            return `<li><span class="act-time">${timeStr}</span><span class="act-dot ${item.color}"></span><span class="act-text">${item.text}</span></li>`;
        }).join('');
    }

    row.innerHTML = `
    <div class="bottom-card">
        <div class="bc-header">
            <span class="bc-title">Energy</span>
            <span class="bc-badge">1/1 fresh</span>
            <button class="bc-btn">Details</button>
        </div>
        <div class="bc-big">${energyKWh.toFixed(2)} <span class="unit">kWh / session</span></div>
        <div class="bc-sub">Estimated from power sampling · ${elapsedH > 0.01 ? (elapsedH*60).toFixed(0) + ' min' : 'just started'}</div>
        <div class="bc-chart"><canvas id="energy-chart"></canvas></div>
        <div class="bc-footer">
            <div class="ff-item"><div class="ff-label">Current</div><div class="ff-val">${sp.toFixed(0)} W</div></div>
            <div class="ff-item"><div class="ff-label">Session</div><div class="ff-val">${energyKWh.toFixed(2)} kWh</div></div>
            <div class="ff-item"><div class="ff-label">Efficiency</div><div class="ff-val">${efficiency} kWh/k</div></div>
        </div>
    </div>

    <div class="bottom-card">
        <div class="bc-header">
            <span class="bc-title">Token totals</span>
            <span class="bc-badge">1 endpoint</span>
            <span class="bc-badge" style="margin-left:4px">All time</span>
            <button class="bc-btn">Details</button>
        </div>
        <div class="bc-big">${fmtToks(genTokens)} <span class="unit">generated</span></div>
        <div class="bc-sub">Counted by splash monitor since tracking began</div>
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
            <tr class="total"><td>TOTAL</td><td>${fmtToks(cachedTokens)}</td><td>${fmtToks(prefillTokens)}</td><td>${fmtToks(genTokens)}</td></tr>
        </table>
    </div>

    <div class="bottom-card">
        <div class="bc-header">
            <span class="bc-title">Recent activity</span>
            <span class="live-badge">LIVE</span>
        </div>
        <ul class="activity-list" id="activity-list">${activityHTML}</ul>
        <button class="bc-btn" style="margin-top:8px;margin-left:0">View all activity</button>
    </div>`;

    // Draw energy chart (synchronously)
    drawEnergyChart();
    setTimeout(() => renderActivity(), 0);
}

function drawEnergyChart() {
    const canvas = document.getElementById('energy-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.parentElement.offsetWidth * 2 || 400;
    const h = canvas.height = 160;
    const pts = history['sys_power'];
    if (!pts || pts.length < 2) { ctx.clearRect(0,0,w,h); return; }
    const recent = pts.slice(-60);
    const mn = 0, mx = Math.max(...recent.map(p=>p[1]), 10);
    const pad = {t:5,r:5,b:5,l:5};
    const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
    ctx.clearRect(0,0,w,h);
    // Bars
    const barW = Math.max(1, cw / recent.length - 1);
    recent.forEach((pt, i) => {
        const x = pad.l + i * (cw / recent.length);
        const bh = (pt[1] / mx) * ch;
        ctx.fillStyle = pt[1] > mx*0.8 ? 'rgba(234,67,53,0.7)' : 'rgba(249,171,0,0.6)';
        ctx.fillRect(x, pad.t + ch - bh, barW, bh);
    });
    // Time labels
    ctx.fillStyle = '#5f6368'; ctx.font = '18px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText(recent[0][0] ? new Date(recent[0][0]).toLocaleTimeString().slice(0,5) : '', pad.l, h - 2);
    ctx.textAlign = 'right';
    ctx.fillText('now', w - pad.r, h - 2);
}

function renderActivity() {
    const list = document.getElementById('activity-list');
    if (!list) return;
    const items = activityLog.slice(-8).reverse();
    if (items.length === 0) {
        list.innerHTML = '<li><span class="act-time">—</span><span class="act-dot blue"></span><span class="act-text">Monitoring started</span></li>';
        return;
    }
    list.innerHTML = items.map(item => {
        const ago = Math.floor((Date.now() - item.t) / 60000);
        const timeStr = ago < 1 ? 'now' : ago < 60 ? ago + 'm' : Math.floor(ago/60) + 'h';
        return `<li><span class="act-time">${timeStr}</span><span class="act-dot ${item.color}"></span><span class="act-text">${item.text}</span></li>`;
    }).join('');
}

function renderCharts() {
    const row = document.getElementById('charts-row');
    row.innerHTML = `
        <div class="chart-card">
            <div class="chart-title"><span>GPU temperature</span><span id="ct-gpu">—</span></div>
            <canvas id="chart-temp" class="chart-canvas"></canvas>
        </div>
        <div class="chart-card">
            <div class="chart-title"><span>Power</span><span id="ct-power">—</span></div>
            <canvas id="chart-power" class="chart-canvas"></canvas>
        </div>
        <div class="chart-card">
            <div class="chart-title"><span>Throughput</span><span id="ct-tp">—</span></div>
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

function updateActivity(splash, macmon) {
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
}

function updateStatus(splash, macmon, ollama) {
    const splashOk = splash.decode_tps !== undefined || splash.mem_current !== undefined;
    const macmonOk = macmon.gpu_temp !== undefined;
    const ollamaOk = !!(ollama && ollama.models !== undefined);
    const el = document.getElementById('header-status');
    const now = new Date().toLocaleTimeString();
    const thermal = splash.thermal_state || 'nominal';
    const ollamaState = ollama && ollama.count > 0 ? `${ollama.count} loaded` : 'idle';
    el.innerHTML = `Splash: <span style="color:${splashOk?'var(--green)':'var(--red)'}">${splashOk?'OK':'FAIL'}</span>
        &nbsp; Ollama: <span style="color:${ollamaOk?'var(--green)':'var(--red)'}">${ollamaOk?ollamaState:'FAIL'}</span>
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
        if (engine === 'ollama' && res.models && res.models.length) {
            btn.textContent = res.models.every(m => m.ok) ? '✓ all released' : '✗ ' + res.models.filter(m=>!m.ok).map(m=>m.model).join(', ');
        }
    }).catch(() => { btn.textContent = '✗ error'; })
    .finally(() => {
        setTimeout(() => { btn.disabled = false; btn.textContent = orig; }, 2500);
    });
}

function fetchData() {
    Promise.all([
        fetch('/api/splash').then(r => r.json()).catch(() => ({})),
        fetch('/api/macmon').then(r => r.json()).catch(() => ({})),
        fetch('/api/ollama').then(r => r.json()).catch(() => ({})),
    ]).then(([splash, macmon, ollama]) => {
        const now = Date.now();
        lastSplash = splash;
        lastMacmon = macmon;
        lastOllama = ollama;

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

        updateActivity(splash, macmon);
        renderNode(splash, macmon, ollama);
        renderStats(splash, macmon);
        renderBottom(splash, macmon);
        renderCharts();
        updateStatus(splash, macmon, ollama);

        setTimeout(fetchData, REFRESH_MS);
    }).catch(() => {
        setTimeout(fetchData, REFRESH_MS);
    });
}

// Initial load
logActivity('Monitoring started', 'green');
logActivity('Mac Mini M5 Pro online', 'green');
fetchData();
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
        return jsonify({"error": "engine must be 'splash' or 'ollama'"}), 400
    except Exception as e:
        return jsonify({"engine": engine, "ok": False, "error": str(e)[:200]}), 502

if __name__ == '__main__':
    print(f"{NODE_NAME} Monitor on 0.0.0.0:{PORT}")
    app.run(host='0.0.0.0', port=PORT, threaded=True)
