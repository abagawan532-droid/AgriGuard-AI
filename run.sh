#!/usr/bin/env bash
echo "======================================================================"
echo "                 AgriGuard AI Platform Starter"
echo "       200 Crops | 200 Tree Species | 200 Pathology Database"
echo "======================================================================"

cd "$(dirname "$0")"

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
else
    PYTHON_EXE="python3"
fi

echo "[INFO] Launching AgriGuard AI on http://127.0.0.1:5000 ..."
$PYTHON_EXE app.py
