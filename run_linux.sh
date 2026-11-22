#!/usr/bin/env bash
# RajaswaMitra - one-command start on Linux / macOS
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "Colleagues on the office network can open: http://${IP:-<this-PC-IP>}:8501"
exec streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501 --browser.gatherUsageStats false
