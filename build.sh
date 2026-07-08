#!/usr/bin/env bash
set -o errexit
pip install --upgrade pip
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt wordnet omw-1.4 punkt_tab
python load_data.py
python preprocess.py
python feature_extraction.py
python train_model.py