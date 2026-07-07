#!/usr/bin/env bash
set -o errexit

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -m nltk.downloader stopwords
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
python -m nltk.downloader punkt_tab

# Retrain models with correct Python version
echo "Training models with correct sklearn version..."
python load_data.py
python preprocess.py
python feature_extraction.py
python train_model.py
echo "Models trained successfully!"
