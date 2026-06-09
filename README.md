# Fake News Detection

**Folder structure**

```
fake_news_detection/
│
├── data/
│   ├── True.csv
│   └── Fake.csv
│
├── models/
│   └── model.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── preprocess.py
├── requirements.txt
└── README.md
```

Quick start

- Install dependencies: `pip install -r requirements.txt`
- Run the app: `python app.py`

Notes

- `data/` should contain the CSV datasets (`True.csv`, `Fake.csv`).
- `models/model.pkl` is the trained model file.
- `templates/index.html` and `static/style.css` are the web UI files.
