import os
import re
import string
import pickle
import numpy as np
import nltk
from flask import (
    Flask,
    render_template,
    request,
    jsonify
)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from urllib.parse import urlparse
from scipy.sparse import hstack, csr_matrix

# ============================================
# DOWNLOAD NLTK
# ============================================

nltk.download('stopwords',  quiet=True)
nltk.download('punkt',      quiet=True)
nltk.download('wordnet',    quiet=True)
nltk.download('omw-1.4',   quiet=True)
nltk.download('punkt_tab', quiet=True)

# ============================================
# INITIALIZE FLASK
# ============================================

app = Flask(__name__)

# ============================================
# INITIALIZE NLP TOOLS
# ============================================

stop_words  = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()

# ============================================
# LOAD MODEL & TOOLS
# ============================================

print("="*50)
print("Loading Model and Tools...")
print("="*50)

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"✅ Model    : {model_info['model_name']}")
print(f"✅ Accuracy : {model_info['accuracy']*100:.2f}%")
print("="*50)


# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text):
    if isinstance(text, float):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
        and len(word) > 2
    ]
    return ' '.join(tokens)


# ============================================
# EXTRACT URL FEATURES
# ============================================

def extract_url_features(url):
    try:
        parsed = urlparse(str(url))
        path_len = len(parsed.path)
    except:
        path_len = 0

    return {
        'url_length'      : len(str(url)),
        'url_dots'        : str(url).count('.'),
        'url_slashes'     : str(url).count('/'),
        'url_has_https'   : int(str(url).startswith('https')),
        'url_path_length' : path_len,
        'url_has_numbers' : int(bool(re.search(r'\d', str(url)))),
        'url_has_special' : int(bool(re.search(r'[!@#$%^&*()_+=]', str(url)))),
        'url_hyphens'     : str(url).count('-')
    }


# ============================================
# EXTRACT DOMAIN FEATURES
# ============================================

def extract_domain_features(domain):
    trusted_domains = [
        'people.com', 'www.dailymail.co.uk',
        'en.wikipedia.org', 'www.usmagazine.com',
        'www.etonline.com', 'www.usatoday.com',
        'www.hollywoodreporter.com', 'variety.com',
        'www.today.com', 'www.nytimes.com',
        'www.bbc.com', 'www.reuters.com',
        'www.theguardian.com', 'www.cnn.com',
        'www.washingtonpost.com'
    ]

    try:
        ext = str(domain).split('.')[-1]
        ext_map = {
            'com':1,'org':2,'net':3,
            'gov':4,'edu':5
        }
        ext_val = ext_map.get(ext, 0)
    except:
        ext_val = 0

    return {
        'domain_length'    : len(str(domain)),
        'domain_trusted'   : int(str(domain) in trusted_domains),
        'domain_numbers'   : int(bool(re.search(r'\d', str(domain)))),
        'domain_has_www'   : int(str(domain).startswith('www')),
        'domain_extension' : ext_val
    }


# ============================================
# EXTRACT TITLE FEATURES
# ============================================

def extract_title_features(title):
    clickbait_words = [
        'breaking', 'shocking', 'exclusive',
        'urgent', 'alert', 'warning', 'secret',
        'revealed', 'exposed', 'incredible',
        'unbelievable', 'must see', 'viral',
        'bombshell', 'scandal', 'explosive',
        'stunning'
    ]

    words = str(title).split()
    text_lower = str(title).lower()

    has_clickbait = int(any(
        w in text_lower for w in clickbait_words
    ))
    clickbait_count = sum(
        1 for w in clickbait_words
        if w in text_lower
    )

    avg_word_len = np.mean([
        len(w) for w in words
    ]) if words else 0

    return {
        'title_word_count'      : len(words),
        'title_char_count'      : len(str(title)),
        'title_has_numbers'     : int(bool(re.search(r'\d', str(title)))),
        'title_uppercase_count' : sum(1 for c in str(title) if c.isupper()),
        'title_exclamation'     : str(title).count('!'),
        'title_question'        : str(title).count('?'),
        'title_avg_word_length' : avg_word_len,
        'title_has_clickbait'   : has_clickbait,
        'title_clickbait_count' : clickbait_count
    }


# ============================================
# EXTRACT TWEET FEATURES
# ============================================

def extract_tweet_features(tweet_num):
    try:
        n = int(tweet_num)
    except:
        n = 0

    if n == 0:       cat = 0
    elif n < 10:     cat = 1
    elif n < 50:     cat = 2
    elif n < 100:    cat = 3
    elif n < 1000:   cat = 4
    else:            cat = 5

    return {
        'tweet_num'      : n,
        'tweet_log'      : np.log1p(n),
        'tweet_category' : cat,
        'is_viral'       : int(n > 1000)
    }


# ============================================
# BUILD FEATURE VECTOR
# ============================================

def build_feature_vector(title, news_url,
                          source_domain, tweet_num):
    # Get all features
    url_feats    = extract_url_features(news_url)
    domain_feats = extract_domain_features(source_domain)
    title_feats  = extract_title_features(title)
    tweet_feats  = extract_tweet_features(tweet_num)

    # Combine all features in correct order
    all_feats = {}
    all_feats.update(url_feats)
    all_feats.update(domain_feats)
    all_feats.update(title_feats)
    all_feats.update(tweet_feats)

    # Build vector in same order as training
    feature_vector = [
        all_feats.get(name, 0)
        for name in feature_names
    ]

    return feature_vector, all_feats



# ============================================
# PREDICT FUNCTION (FIXED)
# ============================================

def predict_news(title, news_url,
                 source_domain, tweet_num):

    print("\n" + "="*50)
    print("DEBUG: Starting Prediction")
    print("="*50)
    print(f"Title         : {title}")
    print(f"News URL      : {news_url}")
    print(f"Source Domain : {source_domain}")
    print(f"Tweet Num     : {tweet_num}")

    # Step 1: Clean title
    cleaned = clean_text(title)
    print(f"\nCleaned Title : {cleaned}")

    # Step 2: TF-IDF
    tfidf_vec = tfidf.transform([cleaned])
    print(f"TF-IDF Shape  : {tfidf_vec.shape}")

    # Step 3: Build features
    try:
        tweet_num = int(tweet_num)
    except:
        tweet_num = 0

    # Auto extract domain from URL
    if news_url and not source_domain:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(news_url)
            source_domain = parsed.netloc
            print(f"Auto Domain   : {source_domain}")
        except:
            source_domain = ''

    # Build all features manually
    url_feats    = extract_url_features(news_url)
    domain_feats = extract_domain_features(source_domain)
    title_feats  = extract_title_features(title)
    tweet_feats  = extract_tweet_features(tweet_num)

    print(f"\nURL Features    : {url_feats}")
    print(f"Domain Features : {domain_feats}")
    print(f"Title Features  : {title_feats}")
    print(f"Tweet Features  : {tweet_feats}")

    # Combine all features
    all_feats = {}
    all_feats.update(url_feats)
    all_feats.update(domain_feats)
    all_feats.update(title_feats)
    all_feats.update(tweet_feats)

    print(f"\nAll Features Keys : {list(all_feats.keys())}")
    print(f"Feature Names     : {feature_names}")

    # Build vector in correct order
    feature_vector = []
    for name in feature_names:
        val = all_feats.get(name, 0)
        feature_vector.append(val)
        print(f"  {name:<30} : {val}")

    print(f"\nFeature Vector : {feature_vector}")
    print(f"Vector Length  : {len(feature_vector)}")

    # Step 4: Scale features
    import numpy as np
    feat_array = np.array(
        feature_vector
    ).reshape(1, -1)

    print(f"\nFeature Array Shape : {feat_array.shape}")
    print(f"Scaler Features     : {scaler.n_features_in_}")

    feat_scaled = scaler.transform(feat_array)
    print(f"Scaled Features     : {feat_scaled}")

    # Step 5: Combine
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([
        tfidf_vec,
        csr_matrix(feat_scaled)
    ])
    print(f"\nCombined Shape : {X_combined.shape}")

    # Step 6: Predict
    prediction = model.predict(X_combined)[0]
    print(f"Raw Prediction : {prediction}")
    print(f"Type           : {type(prediction)}")

    # Step 7: Confidence
    try:
        probability = model.predict_proba(
            X_combined
        )[0]
        confidence = round(max(probability) * 100, 2)
        fake_prob  = round(probability[0] * 100, 2)
        real_prob  = round(probability[1] * 100, 2)
        print(f"Probability : {probability}")
    except Exception as e:
        print(f"predict_proba failed: {e}")
        try:
            scores = model.decision_function(
                X_combined
            )[0]
            confidence = round(
                min(abs(float(scores)) * 10, 100), 2
            )
            if prediction == 1:
                real_prob = confidence
                fake_prob = round(100 - confidence, 2)
            else:
                fake_prob = confidence
                real_prob = round(100 - confidence, 2)
            print(f"Decision Score : {scores}")
        except Exception as e2:
            print(f"decision_function failed: {e2}")
            confidence = 95.0
            fake_prob  = 50.0
            real_prob  = 50.0

    print(f"\nFinal Prediction : {prediction}")
    print(f"Confidence       : {confidence}")
    print(f"Fake Prob        : {fake_prob}")
    print(f"Real Prob        : {real_prob}")
    print("="*50)

    return (
        prediction,
        confidence,
        fake_prob,
        real_prob,
        all_feats
    )

# ============================================
# HOME PAGE
# ============================================

@app.route('/')
def home():
    return render_template(
        'index.html',
        model_name=model_info['model_name'],
        accuracy=round(
            model_info['accuracy'] * 100, 2
        )
    )


# ============================================
# PREDICT ROUTE
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        title         = request.form.get('title', '')
        news_url      = request.form.get('news_url', '')
        source_domain = request.form.get('source_domain', '')
        tweet_num     = request.form.get('tweet_num', 0)

        print(f"\nForm Data Received:")
        print(f"  title         : {title}")
        print(f"  news_url      : {news_url}")
        print(f"  source_domain : {source_domain}")
        print(f"  tweet_num     : {tweet_num}")

        # Validate title
        if not title or title.strip() == "":
            return render_template(
                'index.html',
                error="Please enter a news title!",
                model_name=model_info['model_name'],
                accuracy=round(
                    model_info['accuracy'] * 100, 2
                )
            )

        # Convert tweet_num
        try:
            tweet_num = int(tweet_num)
        except:
            tweet_num = 0

        # Predict
        (prediction,
         confidence,
         fake_prob,
         real_prob,
         features) = predict_news(
            title,
            news_url,
            source_domain,
            tweet_num
        )

        print(f"\nPrediction value : {prediction}")
        print(f"Prediction type  : {type(prediction)}")

        # Convert prediction to int
        prediction = int(prediction)
        print(f"Prediction int   : {prediction}")

        # Set result
        if prediction == 1:
            result       = "REAL NEWS ✅"
            color        = "green"
            bg_color     = "#e8f5e9"
            border_color = "#4CAF50"
        elif prediction == 0:
            result       = "FAKE NEWS ❌"
            color        = "red"
            bg_color     = "#ffebee"
            border_color = "#f44336"
        else:
            result       = f"UNKNOWN ({prediction})"
            color        = "orange"
            bg_color     = "#fff3e0"
            border_color = "#FF9800"

        print(f"Result : {result}")

        return render_template(
            'index.html',
            prediction    = result,
            confidence    = confidence,
            fake_prob     = fake_prob,
            real_prob     = real_prob,
            color         = color,
            bg_color      = bg_color,
            border_color  = border_color,
            title         = title,
            news_url      = news_url,
            source_domain = source_domain,
            tweet_num     = tweet_num,
            features      = features,
            model_name    = model_info['model_name'],
            accuracy      = round(
                model_info['accuracy'] * 100, 2
            )
        )

    except Exception as e:
        import traceback
        print(f"\n❌ ERROR:")
        print(traceback.format_exc())
        return render_template(
            'index.html',
            error=f"Error: {str(e)}",
            model_name=model_info['model_name'],
            accuracy=round(
                model_info['accuracy'] * 100, 2
            )
        )
# ============================================
# API ROUTE
# ============================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        if 'title' not in data:
            return jsonify({
                'error': 'title is required'
            }), 400

        title         = data.get('title', '')
        news_url      = data.get('news_url', '')
        source_domain = data.get('source_domain', '')
        tweet_num     = data.get('tweet_num', 0)

        # Auto extract domain
        if news_url and not source_domain:
            try:
                parsed        = urlparse(news_url)
                source_domain = parsed.netloc
            except:
                source_domain = ''

        (prediction,
         confidence,
         fake_prob,
         real_prob,
         features) = predict_news(
            title, news_url,
            source_domain, tweet_num
        )

        result = "Real News" \
            if prediction == 1 else "Fake News"

        return jsonify({
            'prediction'    : result,
            'confidence'    : confidence,
            'fake_prob'     : fake_prob,
            'real_prob'     : real_prob,
            'model'         : model_info['model_name'],
            'accuracy'      : model_info['accuracy'],
            'features_used' : features
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# RUN APP
# ============================================

if __name__ == '__main__':
    app.run(debug=True)