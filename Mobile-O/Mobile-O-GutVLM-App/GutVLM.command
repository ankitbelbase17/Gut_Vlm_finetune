#!/bin/bash
# Double-click this file in Finder to launch GutVLM.
#
# First run: creates a local Python virtual environment (.venv-mlx) and
# installs requirements-mlx.txt -- takes a minute or two, one time only.
# Every run after that: launches mlx_app.py and opens your browser to it.
#
# Close this Terminal window (or press Ctrl+C) to stop the app.

set -e
cd "$(dirname "$0")"

# MLX only ships arm64-native wheels, and some pinned deps need Python 3.10+.
# The plain `python3` on PATH can't be trusted to satisfy either of those --
# on this exact Mac it resolves to an x86_64 (Rosetta) miniconda build, and
# Apple's own bundled /usr/bin/python3 is 3.9 (too old). Search a list of
# likely candidates and use the first one that's actually arm64 + 3.10+.
find_python() {
    for candidate in python3.13 python3.12 python3.11 python3.10 \
                      /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 \
                      /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.10 \
                      /usr/local/bin/python3.13 /usr/local/bin/python3.12 \
                      /usr/local/bin/python3.11 /usr/local/bin/python3.10 \
                      python3; do
        path=$(command -v "$candidate" 2>/dev/null) || continue
        ok=$("$path" -c "
import platform, sys
print('OK' if platform.machine() == 'arm64' and sys.version_info >= (3, 10) else 'NO')
" 2>/dev/null || true)
        if [ "$ok" = "OK" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

VENV_DIR=".venv-mlx"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    PYTHON_BIN=$(find_python) || {
        echo "Could not find a suitable Python (need a native arm64 build, Python 3.10+)."
        echo "MLX requires Apple Silicon and won't install on an x86_64/Rosetta Python,"
        echo "and some dependencies need Python 3.10 or later."
        echo ""
        echo "Fix: install one via Homebrew (https://brew.sh), then re-run this launcher:"
        echo "    brew install python@3.11"
        read -r -p "Press Enter to close..."
        exit 1
    }
    echo "First run: setting up $VENV_DIR with $PYTHON_BIN ($($PYTHON_BIN --version)) -- this takes a minute or two..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip -q
    pip install -r requirements-mlx.txt
else
    source "$VENV_DIR/bin/activate"
fi

echo "Starting GutVLM..."
python mlx_app.py &
APP_PID=$!
trap 'echo "Stopping GutVLM..."; kill $APP_PID 2>/dev/null' EXIT INT TERM

echo "Waiting for it to come up..."
until curl -s -o /dev/null http://127.0.0.1:7860/ 2>/dev/null; do
    if ! kill -0 $APP_PID 2>/dev/null; then
        echo "GutVLM failed to start -- see the errors above."
        read -r -p "Press Enter to close..."
        exit 1
    fi
    sleep 1
done

open "http://127.0.0.1:7860/"
echo ""
echo "GutVLM is running at http://127.0.0.1:7860/"
echo "Close this window (or press Ctrl+C) to stop it."
wait $APP_PID
