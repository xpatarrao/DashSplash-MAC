#!/usr/bin/env bash
# DashSplash MAC launcher — resuelve el binario Python de forma robusta
# y arranca el monitor desde el repo ~/dashsplash.
# systemd reinicia este script si muere (Restart=always).
set -euo pipefail

REPO="${DASHSPLASH_HOME:-/home/ubuntu/dashsplash}"

# Cargar config local (URLs/IPs reales). `.env` está gitignored; `.env.example` es la plantilla.
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO/.env"
    set +a
fi

PYBIN=$(ls /home/ubuntu/.hermes/tools/python-3.14.7*/bin/python3 2>/dev/null | head -1)
if [ -z "$PYBIN" ]; then
    PYBIN=$(command -v python3)
fi
if [ -z "$PYBIN" ]; then
    echo "ERROR: binario Python no encontrado" >&2
    exit 1
fi

exec "$PYBIN" "$REPO/splash_monitor.py"
