#!/usr/bin/env bash
set -o errexit

<<<<<<< HEAD
# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
=======
echo '==> Installing dependencies...'
pip install --upgrade pip
pip install -r requirements.txt

echo '==> Downloading NLTK data...'
>>>>>>> 23c7146 (Add deployment files: Procfile, runtime.txt, build.sh)
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
python -m nltk.downloader punkt_tab

<<<<<<< HEAD
# Retrain models with correct Python version
echo "Training models with correct sklearn version..."
=======
echo '==> Training models...'
>>>>>>> 23c7146 (Add deployment files: Procfile, runtime.txt, build.sh)
python load_data.py
python preprocess.py
python feature_extraction.py
python train_model.py
<<<<<<< HEAD
echo "Models trained successfully!"
=======

echo '==> Build complete!'
>>>>>>> 23c7146 (Add deployment files: Procfile, runtime.txt, build.sh)
