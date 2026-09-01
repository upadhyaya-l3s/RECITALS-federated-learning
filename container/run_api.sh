#!/bin/bash
# run_api.sh
# ═══════════════════════════════════════════════════════════════════
#  Start the RECITALS FL API on Neumann.
#  Run this every time you want the service to be up.
#
#  Usage:
#    bash run_api.sh           ← foreground (you see logs, Ctrl+C to stop)
#    bash run_api.sh --bg      ← background (keeps running after you log out)
# ═══════════════════════════════════════════════════════════════════

REPO_DIR="$HOME/RECITALS-federated-learning"
SIF="$REPO_DIR/recitals_fl_api.sif"
MODELS_DIR="$REPO_DIR/models"
PORT=8000
LOGFILE="$REPO_DIR/api_server.log"

# ── Checks ────────────────────────────────────────────────────────
if [ ! -f "$SIF" ]; then
    echo "✗ Image not found: $SIF"
    echo "  Run setup_neumann.sh first."
    exit 1
fi

mkdir -p "$MODELS_DIR"

# Check if model weights exist and warn if not
if [ ! -f "$MODELS_DIR/federated_lstm_heterogeneous_moderate.pt" ]; then
    echo "⚠ WARNING: No model weights found at $MODELS_DIR/federated_lstm_heterogeneous_moderate.pt"
    echo "  The API will run in stub/heuristic mode."
    echo "  Run save_model.py after training to load real weights."
    echo ""
fi

# ── Run ──────────────────────────────────────────────────────────
if [ "$1" == "--bg" ]; then
    # Background mode: keeps running after SSH session ends
    echo "Starting API in background mode..."
    echo "Logs → $LOGFILE"
    echo "To stop: kill \$(cat $REPO_DIR/api_server.pid)"
    nohup apptainer run \
        --bind "$MODELS_DIR:/app/models" \
        "$SIF" \
        > "$LOGFILE" 2>&1 &
    echo $! > "$REPO_DIR/api_server.pid"
    echo "✓ Started (PID: $(cat $REPO_DIR/api_server.pid))"
    echo ""
    echo "Access via SSH tunnel on your laptop:"
    echo "  ssh -L $PORT:localhost:$PORT upadhyaya@rambo.l3s.de"
    echo "  Then open: http://localhost:$PORT/docs"
else
    # Foreground mode: you see logs live, Ctrl+C to stop
    echo "Starting RECITALS FL API on port $PORT..."
    echo "Press Ctrl+C to stop."
    echo ""
    echo "Access via SSH tunnel on your laptop:"
    echo "  ssh -L $PORT:localhost:$PORT upadhyaya@rambo.l3s.de"
    echo "  Then open: http://localhost:$PORT/docs"
    echo ""
    apptainer run \
        --bind "$MODELS_DIR:/app/models" \
        "$SIF"
fi
