# DashSplash MAC

<p align="center">
  <img src="preview.jpg" alt="DashSplash MAC dashboard" width="700">
</p>

<p align="center">
  <a href="README.md">Español</a> · <strong>English</strong>
</p>

Real-time monitoring dashboard for **Splash** (LLM inference engine on Apple Silicon) + **macmon** (GPU/CPU/fan hardware metrics) + **Ollama** (loaded models, optional).

> Style: a fork of [sparkDash](https://github.com/MiaAI-Lab/sparkDash) adapted to a **single machine** (Mac Mini M5 Pro).
>
> **Ollama is optional**: if its endpoint doesn't respond, the dashboard omits it and keeps working with Splash + macmon.

## What it shows

| Section | Source | Metrics |
|---|---|---|
| **Header** | Splash + macmon + Ollama | OK/FAIL status, `Ollama: N loaded`, thermal (nominal/fair/serious/critical), runtime, clock |
| **Node card** | Splash + macmon + Ollama | Unified memory with **per-engine breakdown** (bar: splash/ollama/system/free + GB each), GPU util %, temp, power, freq, decode/prefill tok/s, scheduler, splash model, ollama models, TTFT/ITL |
| **Aggregated stats** | Splash + macmon | Decode tok/s (sparkline), power W (sparkline), memory in use % (real system RAM from macmon: splash + ollama + OS), total tokens |
| **Energy** | macmon | Session kWh, current W, efficiency kWh/token, 24h bar chart |
| **Token totals** | Splash | Generated / Cached / Prefill, per-model table |
| **Activity** | Splash + macmon + Ollama | Live log (decode started/finished, thermal, GPU hot, Ollama models loaded/unloaded) |
| **Charts** | Splash + macmon | GPU temp, power, throughput (5 min lines) |

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Mac Mini M5 Pro (192.168.1.x)                                         │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  Splash      │    │  macmon      │    │  Ollama      │               │
│  │  :8000       │    │  :9090       │    │  :11434      │               │
│  │  /metrics    │    │  /json       │    │  /api/ps     │               │
│  │  (Prometheus)│    │  (JSON)      │    │  (JSON)      │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
└─────────┼───────────────────┼───────────────────┼───────────────────────┘
          │                   │                   │
          │      HTTP GET (5s poll, 3 sources)   │
          ▼                   ▼                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  VM Linux (192.168.1.x)                                                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │  splash_monitor.py (Flask, :8081)                           │        │
│  │                                                            │        │
│  │  /            → dashboard HTML (template string)            │        │
│  │  /api/splash  → Splash metrics (JSON)                      │        │
│  │  /api/macmon  → macmon metrics (real RAM, GPU, fans)       │        │
│  │  /api/ollama  → loaded Ollama models (JSON)                │        │
│  │                                                            │        │
│  │  update_loop() → polls every 5s, stores in dicts           │        │
│  │  JS fetch     → /api/* every 5s, draws canvas              │        │
│  └──────────────────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────────────────┘
          │
          │  http://192.168.1.x:8081
          ▼
     Browser (any machine on the LAN)
```

### Data flow

1. **Backend** (Python/Flask): `update_loop()` runs `requests.get` against the **3 sources** every 5s: Splash `/metrics` (Prometheus), macmon `/json` (real system RAM, GPU, fans) and Ollama `/api/ps` (loaded models), parses them and stores them in in-memory dicts. Ollama is **optional**: if its endpoint doesn't respond, the UI omits it without breaking anything.
2. **Frontend** (vanilla JS): `fetchData()` calls `/api/splash`, `/api/macmon` and `/api/ollama` every 5s, accumulates into `history[]` (last hour, 720-point cap), and redraws the whole DOM + canvas. Unified memory is shown with a **per-engine breakdown** (splash/ollama/system/free).
3. Charts use **2D canvas** (no external libraries): sparklines, energy bars, temp/power/throughput lines.

## Required on the Mac

- **Splash** running → exposes `:8000/metrics`
- **macmon** running → exposes `:9090/json`
- **Ollama** running (optional) → exposes `:11434` with `OLLAMA_HOST=0.0.0.0` (so the VM can reach it over the network)

```bash
# On the Mac, if you don't have them:
brew install vladkens/macmon/macmon   # macmon
brew install ollama                   # Ollama (optional)
brew services start ollama            # or: ollama serve
# Splash: install per its README
```

**Ollama over the network** (only if the VM queries it): it must listen on all interfaces.
```bash
OLLAMA_HOST=0.0.0.0 ollama serve     # instead of 'ollama serve' (which listens on 127.0.0.1)
```

## Installation on the VM

### Requirements
- Python 3.10+ with `flask` and `requests`
- systemd (Linux)

### 1. Copy the repo

```bash
git clone <URL> ~/dashsplash
# or: rsync -av dashsplash/ user@192.168.1.x:~/dashsplash/
```

### 2. Install the Python dependencies

```bash
# If you use the Hermes Python:
pip install flask requests
# or create a venv:
python3 -m venv ~/dashsplash/.venv
~/dashsplash/.venv/bin/pip install flask requests
```

### 3. Install the systemd service

```bash
bash install.sh
```

### 4. Verify

```bash
systemctl status dashsplash
curl http://localhost:8081/
```

### 5. Open in a browser

```
http://192.168.1.x:8081
```

## Configuration

Everything is configured with **environment variables** (or a `.env` file that the launcher loads). The repo is 100% generic: it contains no personal IPs or URLs.

1. **Copy the template** (it is gitignored, not pushed):

   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`** with your values:

   ```bash
   # .env — your local configuration (gitignored, never pushed to GitHub)
   DASHSPLASH_SPLASH_URL="http://192.168.1.x:8000/metrics"
   DASHSPLASH_MACMON_URL="http://192.168.1.x:9090/json"
   DASHSPLASH_OLLAMA_URL="http://192.168.1.x:11434/api/ps"   # optional (loaded models)
   DASHSPLASH_MODEL="Qwen3.8-27B-4bit"
   DASHSPLASH_GPU_TDP="100"          # GPU TDP in watts
   DASHSPLASH_MEM_GB="64"            # total unified memory in GB (cosmetic title only)
   DASHSPLASH_NODE_NAME="Mac Mini M5 Pro"
   DASHSPLASH_NODE_IP="192.168.1.x"  # cosmetic badge only
   DASHSPLASH_PORT="8081"
   ```

3. **Restart** the service to pick up the changes:

   ```bash
   systemctl restart dashsplash
   ```

Without a `.env`, the dashboard starts with default values (`127.0.0.1`) — handy for testing.

## Files

| File | Description |
|---|---|
| `splash_monitor.py` | Flask server + the dashboard HTML/JS (~880 lines) |
| `dashsplash.sh` | Launcher: resolves the Python binary and starts the monitor |
| `systemd/dashsplash.service` | systemd service (starts on boot, `Restart=always`) |
| `install.sh` | Install script (copies the service, enables, starts) |
| `README.md` | This documentation (Spanish) / `README.en.md` (English) |

## Available metrics

### Splash (Prometheus)

| Metric | Description |
|---|---|
| `splash_memory_current_bytes` | Unified RAM in use (only what splash sees; does **not** include ollama) |
| `splash_memory_limit_bytes` | Memory ceiling splash is allowed to use (≈ `MEMORY_TOTAL_GB`), **not** the system's total RAM |
| `splash_memory_headroom_bytes` | Headroom within splash's limit |
| `splash_memory_peak_bytes` | Peak |
| `splash_thermal_state.{nominal,fair,serious,critical}` | Thermal state |
| `splash_decode_tokens_per_second` | Decode throughput |
| `splash_prefill_tokens_per_second` | Prefill throughput |
| `splash_ttft_p50/p95_milliseconds` | Time to first token |
| `splash_itl_p50/p95_milliseconds` | Inter-token latency |
| `splash_scheduler_decoding` | Jobs decoding |
| `splash_scheduler_prefilling` | Jobs in prefill |
| `splash_scheduler_queued` | Jobs in queue |
| `splash_kv_pages_allocated/active/cache` | KV cache |
| `splash_kv_allocated_bytes` | KV in bytes |
| `splash_requests_submitted/completed/failed/cancelled_total` | Requests |
| `splash_prefill_input_tokens_total` | Cumulative prefill tokens |
| `splash_decode_output_tokens_total` | Cumulative generated tokens |
| `splash_drafted_tokens_total` | Draft tokens (speculative decoding) |
| `splash_accepted_draft_tokens_total` | Accepted drafts |
| `splash_cache_hits/cold_misses/reused_tokens_total` | Cache |

### macmon (JSON)

| Field | Description |
|---|---|
| `temp.gpu_temp_avg` | GPU temp °C |
| `temp.cpu_temp_avg` | CPU temp °C |
| `gpu_power` | GPU power W |
| `cpu_power` | CPU power W |
| `sys_power` | System power W |
| `gpu_usage` | GPU utilization (0-1 array) |
| `gpu_freq_mhz` | GPU frequency MHz |
| `cpu_usage_pct` | CPU utilization % |
| `fans[0].rpm` | Fan RPM |
| `memory.ram_total` | **Total system RAM** (bytes) — the source of truth for free memory |
| `memory.ram_usage` | System RAM used (bytes) — includes splash + ollama + OS |
| `memory.swap_total` / `memory.swap_usage` | Swap total/used (bytes) |

### Ollama (JSON `/api/ps`)

> **Optional**: if the endpoint doesn't respond, the UI hides its block and breaks nothing.

| Field | Description |
|---|---|
| `models[].name` | Name of a loaded model (e.g. `qwen3.5:4b-mlx`) |
| `models[].size` | Memory in use (bytes) — on a unified Mac this is the model's real RAM |
| `models[].size_vram` | The same memory viewed as GPU (unified pool) — **do not add to `size`** (double counts) |
| `models[].expires_at` | When it is unloaded from RAM (inactivity) |

**Memory note (Apple Silicon)**: on a Mac with unified memory, Ollama reports the model in both `size` **and** `size_vram` with the same value (it is a single pool). The dashboard counts only `size` for the Ollama segment. Total memory bar = splash + ollama + system (remainder) + free.

## Auto-restart and persistence

The systemd service with `Restart=always` + `RestartSec=3` guarantees:
- **Start on boot** (`WantedBy=multi-user.target`)
- **Restart if the process dies** (crash, OOM, etc.)
- **Restart if the Flask server hangs** (systemd watchdog monitoring)

The bash `while true` wrapper from `/tmp/run_splash_monitor.sh` is no longer needed.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| 500 Internal Server Error | DASHBOARD is a tuple (comma after `'''`) | Remove the comma: `'''` not `''',` |
| Empty charts | Canvas `offsetWidth` = 0 at draw time | Use `requestAnimationFrame` or `setTimeout(0)` |
| "Splash FAIL" | Splash not responding | Check `curl http://192.168.1.x:8000/metrics` |
| "Macmon FAIL" | macmon not responding | Check `curl http://192.168.1.x:9090/json` |
| Ollama missing from the dashboard | Endpoint not responding / listening on 127.0.0.1 | Start Ollama with `OLLAMA_HOST=0.0.0.0` and check `DASHSPLASH_OLLAMA_URL` |
| Double-counted Ollama memory | Adding `size` + `size_vram` (unified pool) | Count only `size` (see Ollama section) |
| Port 8081 in use | Zombie process | `fuser -k 8081/tcp` |
| Port blocked by firewall | iptables | `sudo iptables -I INPUT -p tcp --dport 8081 -j ACCEPT` |

## License

MIT
