#!/usr/bin/env bash
# DashSplash MAC — instalador
# Copia los archivos al deploy dir, instala systemd, habilita y arranca.
# Uso: sudo bash install.sh   (desde la raíz del repo)
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_HOME="${DASHSPLASH_HOME:-/home/ubuntu}"
SVC_NAME="dashsplash"
SVC_FILE="/etc/systemd/system/${SVC_NAME}.service"

echo "==> DashSplash MAC installer"
echo "    Repo:     $REPO"
echo "    Deploy:   $DEPLOY_HOME"

# 1. Cópia el launcher al home (referenciado por el servicio)
echo "==> Copiando launcher a $DEPLOY_HOME/dashsplash.sh"
install -m 755 "$REPO/dashsplash.sh" "$DEPLOY_HOME/dashsplash.sh"

# 2. Si el repo está en otro sitio que $DEPLOY_HOME, el launcher usa DASHSPLASH_HOME
#    Pero si el repo ya está en $DEPLOY_HOME/dashsplash, es directo.
if [ "$REPO" = "$DEPLOY_HOME/dashsplash" ]; then
    echo "    (repo en $DEPLOY_HOME/dashsplash — launcher lo encuentra)"
else
    echo "    (repo en otro sitio: el servicio fijará DASHSPLASH_HOME=$REPO)"
fi

# 3. Servicio systemd
echo "==> Instalando servicio $SVC_NAME.service"
if [ "$REPO" = "$DEPLOY_HOME/dashsplash" ]; then
    ENV_LINE=""
else
    ENV_LINE="Environment=DASHSPLASH_HOME=$REPO"
fi

sudo tee "$SVC_FILE" > /dev/null << EOF
[Unit]
Description=DashSplash MAC - Mac Mini M5 Pro Splash+Macmon monitor dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$DEPLOY_HOME
ExecStart=$DEPLOY_HOME/dashsplash.sh
${ENV_LINE}
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SVC_NAME" 2>&1 | sed 's/^/    /'

# 4. Reinicia si ya estaba activo, arranca si no
if systemctl is-active --quiet "$SVC_NAME" 2>/dev/null; then
    echo "==> Reiniciando servicio existente"
    sudo systemctl restart "$SVC_NAME"
else
    echo "==> Arrancando servicio"
    sudo systemctl start "$SVC_NAME"
fi

sleep 3
echo "==> Estado: $(systemctl is-active "$SVC_NAME")"
echo ""
echo "✓ Instalación completa."
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):8081"
echo "  Ver logs:  journalctl -u $SVC_NAME -f"
