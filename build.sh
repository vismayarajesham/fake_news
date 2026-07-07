#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

python -m nltk.downloader stopwords
python -m nltk.downloader punkt
python -m nltk.downloader wordnet
python -m nltk.downloader omw-1.4
python -m nltk.downloader punkt_tab
