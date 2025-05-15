#!/bin/bash
echo "--- Running SIMPLIFIED startup.sh ---"
echo "PORT environment variable: $PORT"
echo "Starting Gunicorn directly..."
# Make sure 'antenv' is the virtual environment Oryx creates, or adjust path if needed
# The path to gunicorn might be directly available if the venv is activated by Oryx.
# /tmp/8dd93a99a32d78b/antenv/bin/gunicorn # This was from previous logs, might change

# Try with the gunicorn that should be in the PATH after Oryx activates the venv
gunicorn --bind=0.0.0.0:${PORT:-8000} --workers=1 --threads=2 --timeout=300 --access-logfile '-' --error-logfile '-' --log-level debug app:app

echo "--- SIMPLIFIED startup.sh finished (or Gunicorn took over) ---"
