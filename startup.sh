#!/binbash

# Create instance and uploads directory if they don't exist (app.py also attempts this)
# These paths are relative to the app's root directory.
mkdir -p instance
mkdir -p uploads
mkdir -p nltk_data_local # Ensure base NLTK data dir for local downloads exists

# (Optional but recommended for Azure App Service cold starts)
# Download NLTK 'punkt' during startup if not bundled or if the app's internal download fails.
# The app.py script also tries to download it, but this can be a fallback or primary method.
# Ensure the `download_dir` matches what `app.py` expects or add it to `nltk.data.path`.
echo "Attempting to ensure NLTK 'punkt' tokenizer is available..."
python -c "import nltk; import os; nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data_local'); nltk.download('punkt', download_dir=nltk_data_dir, quiet=True, raise_on_error=False); print(f'NLTK download attempted to {nltk_data_dir}')"

# Initialize Database (Flask-SQLAlchemy's db.create_all() in app.py should handle this on app start)
# If you use Flask-Migrate, you would run migrations here:
# flask db upgrade

# Start Gunicorn
# Use environment variables for port for Azure compatibility.
# Timeout is important for potentially long-running Azure AI tasks.
# Workers and threads can be adjusted based on your App Service Plan.
echo "Starting Gunicorn..."
gunicorn --bind=0.0.0.0:${PORT:-8000} \
         --workers=${GUNICORN_WORKERS:-2} \
         --threads=${GUNICORN_THREADS:-2} \
         --timeout ${GUNICORN_TIMEOUT:-300} \
         --access-logfile '-' \
         --error-logfile '-' \
         --log-level info \
         app:app

echo "Gunicorn started."
