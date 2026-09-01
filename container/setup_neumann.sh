#!/bin/bash
# setup_neumann.sh
# ═══════════════════════════════════════════════════════════════════
#  Run this script on Neumann to set up the full RECITALS FL API.
#
#  Usage:
#    bash setup_neumann.sh
#
#  What it does, in order:
#    1. Goes to your repo
#    2. Creates the folder structure
#    3. Builds the Apptainer image (takes ~5-10 min first time)
#    4. Tells you how to run it
# ═══════════════════════════════════════════════════════════════════

set -e   # stop immediately if any command fails

REPO_DIR="$HOME/recitals/PPFL/RECITALS-federated-learning"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  RECITALS FL API – Neumann Setup"
echo "════════════════════════════════════════════════════════════"

# ── Step 1: Go to repo ───────────────────────────────────────────
echo ""
echo "[1/5] Navigating to repo..."
if [ ! -d "$REPO_DIR" ]; then
    echo "  Repo not found at $REPO_DIR — cloning..."
    cd "$HOME"
    git clone https://github.com/upadhyaya-l3s/RECITALS-federated-learning.git
fi
cd "$REPO_DIR"
echo "  ✓ In: $(pwd)"

# ── Step 2: Create folder structure ──────────────────────────────
echo ""
echo "[2/5] Creating folder structure..."
mkdir -p api/app models
echo "  ✓ Created: api/app/  and  models/"

# ── Step 3: Check that the API files exist ────────────────────────
echo ""
echo "[3/5] Checking API files..."
if [ ! -f "api/app/main.py" ]; then
    echo "  ✗ ERROR: api/app/main.py not found!"
    echo "    Copy it from the files Claude gave you, then re-run this script."
    exit 1
fi
if [ ! -f "recitals_fl_api.def" ]; then
    echo "  ✗ ERROR: recitals_fl_api.def not found!"
    echo "    Copy it from the files Claude gave you, then re-run this script."
    exit 1
fi
echo "  ✓ api/app/main.py found"
echo "  ✓ recitals_fl_api.def found"

# ── Step 4: Build the Apptainer image ────────────────────────────
echo ""
echo "[4/5] Building Apptainer image (this takes 5-10 minutes)..."
echo "      It downloads Python 3.11 + PyTorch + FastAPI inside the image."
echo "      You only need to do this once (or after code changes)."
echo ""

# APPTAINER_TMPDIR: use your home dir for temp files during build
# (avoids /tmp running out of space on shared servers)
export APPTAINER_TMPDIR="$HOME/.apptainer_tmp"
mkdir -p "$APPTAINER_TMPDIR"

apptainer build --force recitals_fl_api.sif recitals_fl_api.def

echo ""
echo "  ✓ Image built: $(pwd)/recitals_fl_api.sif"
echo "  ✓ Size: $(du -sh recitals_fl_api.sif | cut -f1)"

# ── Step 5: Print run instructions ───────────────────────────────
echo ""
echo "[5/5] Setup complete! Here is how to use it:"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  TO RUN THE API:"
echo ""
echo "    cd $REPO_DIR"
echo "    apptainer run --bind ./models:/app/models recitals_fl_api.sif"
echo ""
echo "  THEN (from your laptop) open an SSH tunnel:"
echo ""
echo "    ssh -L 8000:localhost:8000 upadhyaya@rambo.l3s.de"
echo ""
echo "  THEN open in your browser:"
echo "    http://localhost:8000/docs      ← interactive Swagger UI"
echo "    http://localhost:8000/health    ← health check"
echo ""
echo "  TO STOP: press Ctrl+C in the terminal where it's running"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  NOTE: Model weights not yet loaded (stub mode)."
echo "  Run save_model.py after training to enable real inference."
echo ""
