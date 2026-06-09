import pandas as pd
import numpy as np
import re
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from urllib.parse import urlparse

# ============================================
# DOWNLOAD NLTK RESOURCES
# ============================================

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

# ============================================
# INITIALIZE
# ============================================

stop_words   = set(stopwords.words('english'))
lemmatizer   = WordNetLemmatizer()
os.makedirs('data',   exist_ok=True)
os.makedirs('static', exist_ok=True)

# ============================================
# STEP 1: CLEAN DATASET
# ============================================

def clean_dataset(df):
    print("STEP 1: Cleaning Dataset...")
    print("="*50)

    # Remove duplicates
    print(f"Before removing duplicates : {len(df)}")
    df = df.drop_duplicates()
    print(f"After removing duplicates  : {len(df)}")

    # Drop rows where title is missing
    print(f"Before dropping NA title   : {len(df)}")
    df = df.dropna(subset=['title', 'real'])
    print(f"After dropping NA title    : {len(df)}")

    # Fill missing news_url
    df['news_url'] = df['news_url'].fillna('unknown')

    # Fill missing source_domain
    df['source_domain'] = df['source_domain'].fillna('unknown')

    # Fill missing tweet_num
    df['tweet_num'] = df['tweet_num'].fillna(0)

    # Ensure correct types
    df['title']         = df['title'].astype(str)
    df['news_url']      = df['news_url'].astype(str)
    df['source_domain'] = df['source_domain'].astype(str)
    df['tweet_num']     = df['tweet_num'].astype(int)
    df['real']          = df['real'].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    print(f"\nFinal Dataset Size : {len(df)}")
    print(f"Real News (1)      : {len(df[df['real']==1])}")
    print(f"Fake News (0)      : {len(df[df['real']==0])}")

    return df


# ============================================
# STEP 2: CLEAN TITLE TEXT
# ============================================

def clean_text(text):
    """
    Clean title text
    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation
    5. Remove numbers
    6. Remove extra whitespace
    7. Tokenize
    8. Remove stopwords
    9. Lemmatize
    """

    if isinstance(text, float):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # 5. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 6. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Tokenize
    tokens = word_tokenize(text)

    # 8. Remove stopwords + lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
        and len(word) > 2
    ]

    return ' '.join(tokens)


# ============================================
# STEP 3: EXTRACT URL FEATURES
# ============================================

def extract_url_features(df):
    """
    Extract features from news_url:
    1. URL length
    2. Number of dots in URL
    3. Number of slashes in URL
    4. Has HTTPS (secure)
    5. URL path length
    6. Has numbers in URL
    7. Has special chars in URL
    """
    print("\nSTEP 3: Extracting URL Features...")
    print("="*50)

    def url_length(url):
        return len(str(url))

    def count_dots(url):
        return str(url).count('.')

    def count_slashes(url):
        return str(url).count('/')

    def has_https(url):
        return int(str(url).startswith('https'))

    def url_path_length(url):
        try:
            parsed = urlparse(str(url))
            return len(parsed.path)
        except:
            return 0

    def has_numbers_in_url(url):
        return int(bool(re.search(r'\d', str(url))))

    def has_special_chars(url):
        return int(bool(
            re.search(r'[!@#$%^&*()_+=]', str(url))
        ))

    def count_hyphens(url):
        return str(url).count('-')

    df['url_length']        = df['news_url'].apply(url_length)
    df['url_dots']          = df['news_url'].apply(count_dots)
    df['url_slashes']       = df['news_url'].apply(count_slashes)
    df['url_has_https']     = df['news_url'].apply(has_https)
    df['url_path_length']   = df['news_url'].apply(url_path_length)
    df['url_has_numbers']   = df['news_url'].apply(has_numbers_in_url)
    df['url_has_special']   = df['news_url'].apply(has_special_chars)
    df['url_hyphens']       = df['news_url'].apply(count_hyphens)

    print("URL Features Extracted:")
    url_cols = [
        'url_length', 'url_dots', 'url_slashes',
        'url_has_https', 'url_path_length',
        'url_has_numbers', 'url_has_special',
        'url_hyphens'
    ]
    print(df[url_cols].describe())

    return df


# ============================================
# STEP 4: EXTRACT DOMAIN FEATURES
# ============================================

def extract_domain_features(df):
    """
    Extract features from source_domain:
    1. Domain length
    2. Is known trusted domain
    3. Domain has numbers
    4. Domain level (www vs non-www)
    """
    print("\nSTEP 4: Extracting Domain Features...")
    print("="*50)

    # Known trusted domains list
    trusted_domains = [
        'people.com',
        'www.dailymail.co.uk',
        'en.wikipedia.org',
        'www.usmagazine.com',
        'www.etonline.com',
        'www.usatoday.com',
        'www.hollywoodreporter.com',
        'variety.com',
        'www.today.com',
        'www.nytimes.com',
        'www.bbc.com',
        'www.reuters.com',
        'www.theguardian.com',
        'www.cnn.com',
        'www.washingtonpost.com'
    ]

    def domain_length(domain):
        return len(str(domain))

    def is_trusted(domain):
        return int(str(domain) in trusted_domains)

    def domain_has_numbers(domain):
        return int(bool(re.search(r'\d', str(domain))))

    def has_www(domain):
        return int(str(domain).startswith('www'))

    def domain_extension(domain):
        try:
            ext = str(domain).split('.')[-1]
            if ext == 'com':  return 1
            if ext == 'org':  return 2
            if ext == 'net':  return 3
            if ext == 'gov':  return 4
            if ext == 'edu':  return 5
            return 0
        except:
            return 0

    df['domain_length']    = df['source_domain'].apply(domain_length)
    df['domain_trusted']   = df['source_domain'].apply(is_trusted)
    df['domain_numbers']   = df['source_domain'].apply(domain_has_numbers)
    df['domain_has_www']   = df['source_domain'].apply(has_www)
    df['domain_extension'] = df['source_domain'].apply(domain_extension)

    print("Domain Features Extracted:")
    domain_cols = [
        'domain_length', 'domain_trusted',
        'domain_numbers', 'domain_has_www',
        'domain_extension'
    ]
    print(df[domain_cols].describe())

    # Trusted domain vs label
    print(f"\nTrusted Domain vs Label:")
    print(pd.crosstab(
        df['domain_trusted'],
        df['real'],
        rownames=['Trusted'],
        colnames=['Real']
    ))

    return df


# ============================================
# STEP 5: EXTRACT TITLE FEATURES
# ============================================

def extract_title_features(df):
    """
    Extract features from title:
    1. Title word count
    2. Title char count
    3. Title has numbers
    4. Title uppercase count
    5. Title exclamation count
    6. Title question mark count
    7. Title avg word length
    8. Title has clickbait words
    """
    print("\nSTEP 5: Extracting Title Features...")
    print("="*50)

    # Clickbait words
    clickbait_words = [
        'breaking', 'shocking', 'exclusive',
        'urgent', 'alert', 'warning', 'secret',
        'revealed', 'exposed', 'incredible',
        'unbelievable', 'you won\'t believe',
        'must see', 'viral', 'bombshell',
        'scandal', 'explosive', 'stunning'
    ]

    def word_count(text):
        return len(str(text).split())

    def char_count(text):
        return len(str(text))

    def has_numbers(text):
        return int(bool(re.search(r'\d', str(text))))

    def uppercase_count(text):
        return sum(1 for c in str(text) if c.isupper())

    def exclamation_count(text):
        return str(text).count('!')

    def question_count(text):
        return str(text).count('?')

    def avg_word_length(text):
        words = str(text).split()
        if len(words) == 0:
            return 0
        return np.mean([len(w) for w in words])

    def has_clickbait(text):
        text_lower = str(text).lower()
        return int(any(
            word in text_lower
            for word in clickbait_words
        ))

    def count_clickbait_words(text):
        text_lower = str(text).lower()
        return sum(
            1 for word in clickbait_words
            if word in text_lower
        )

    df['title_word_count']       = df['title'].apply(word_count)
    df['title_char_count']       = df['title'].apply(char_count)
    df['title_has_numbers']      = df['title'].apply(has_numbers)
    df['title_uppercase_count']  = df['title'].apply(uppercase_count)
    df['title_exclamation']      = df['title'].apply(exclamation_count)
    df['title_question']         = df['title'].apply(question_count)
    df['title_avg_word_length']  = df['title'].apply(avg_word_length)
    df['title_has_clickbait']    = df['title'].apply(has_clickbait)
    df['title_clickbait_count']  = df['title'].apply(count_clickbait_words)

    print("Title Features Extracted:")
    title_cols = [
        'title_word_count', 'title_char_count',
        'title_has_numbers', 'title_uppercase_count',
        'title_exclamation', 'title_question',
        'title_avg_word_length', 'title_has_clickbait',
        'title_clickbait_count'
    ]
    print(df[title_cols].describe())

    # Clickbait vs label
    print(f"\nClickbait Words vs Label:")
    print(pd.crosstab(
        df['title_has_clickbait'],
        df['real'],
        rownames=['Has Clickbait'],
        colnames=['Real']
    ))

    return df


# ============================================
# STEP 6: TWEET FEATURES
# ============================================

def extract_tweet_features(df):
    """
    Extract features from tweet_num:
    1. Tweet count categories
    2. Log tweet count
    3. Is viral (tweet > threshold)
    """
    print("\nSTEP 6: Extracting Tweet Features...")
    print("="*50)

    # Log tweet count (handle 0)
    df['tweet_log'] = np.log1p(df['tweet_num'])

    # Tweet categories
    def tweet_category(n):
        if n == 0:    return 0  # No tweets
        if n < 10:    return 1  # Low
        if n < 50:    return 2  # Medium
        if n < 100:   return 3  # High
        if n < 1000:  return 4  # Very High
        return 5                # Viral

    df['tweet_category'] = df['tweet_num'].apply(
        tweet_category
    )

    # Is viral
    df['is_viral'] = (df['tweet_num'] > 1000).astype(int)

    print("Tweet Features:")
    print(df[[
        'tweet_num', 'tweet_log',
        'tweet_category', 'is_viral'
    ]].describe())

    print(f"\nTweet Category vs Label:")
    print(pd.crosstab(
        df['tweet_category'],
        df['real'],
        rownames=['Tweet Category'],
        colnames=['Real']
    ))

    return df


# ============================================
# STEP 7: HANDLE CLASS IMBALANCE
# ============================================

def handle_imbalance(df):
    """
    Handle class imbalance:
    Real: 75.19%
    Fake: 24.81%

    Options:
    1. Use class_weight='balanced' in model
    2. Undersample majority class
    3. SMOTE oversampling (after vectorization)
    """
    print("\nSTEP 7: Handling Class Imbalance...")
    print("="*50)

    real_count = len(df[df['real'] == 1])
    fake_count = len(df[df['real'] == 0])

    print(f"Before balancing:")
    print(f"  Real News : {real_count}")
    print(f"  Fake News : {fake_count}")
    print(f"  Ratio     : {real_count/fake_count:.2f}:1")

    # Option: Undersample majority class
    real_df = df[df['real'] == 1].sample(
        n=fake_count * 2,
        random_state=42
    )
    fake_df = df[df['real'] == 0]

    # Combine
    df_balanced = pd.concat(
        [real_df, fake_df],
        axis=0
    ).reset_index(drop=True)

    # Shuffle
    df_balanced = df_balanced.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print(f"\nAfter balancing:")
    print(f"  Real News : {len(df_balanced[df_balanced['real']==1])}")
    print(f"  Fake News : {len(df_balanced[df_balanced['real']==0])}")

    return df_balanced


# ============================================
# STEP 8: WORD CLOUD
# ============================================

def generate_wordcloud(df):
    print("\nSTEP 8: Generating Word Clouds...")

    real_text = ' '.join(
        df[df['real']==1]['cleaned_title'].tolist()
    )
    fake_text = ' '.join(
        df[df['real']==0]['cleaned_title'].tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Real News
    wc_real = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Greens',
        max_words=100
    ).generate(real_text)
    axes[0].imshow(wc_real, interpolation='bilinear')
    axes[0].set_title(
        'Real News - Common Words',
        fontsize=13,
        fontweight='bold'
    )
    axes[0].axis('off')

    # Fake News
    wc_fake = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='Reds',
        max_words=100
    ).generate(fake_text)
    axes[1].imshow(wc_fake, interpolation='bilinear')
    axes[1].set_title(
        'Fake News - Common Words',
        fontsize=13,
        fontweight='bold'
    )
    axes[1].axis('off')

    plt.suptitle(
        'Word Cloud Analysis',
        fontsize=15,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('static/wordcloud.png')
    plt.show()
    print("Saved to static/wordcloud.png")


# ============================================
# STEP 9: VISUALIZE FEATURES
# ============================================

def visualize_features(df):
    print("\nSTEP 9: Visualizing Features...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Feature Analysis',
        fontsize=16,
        fontweight='bold'
    )

    # Plot 1: Tweet count vs label
    df[df['real']==1]['tweet_log'].hist(
        bins=30, color='green',
        alpha=0.7, label='Real',
        ax=axes[0,0]
    )
    df[df['real']==0]['tweet_log'].hist(
        bins=30, color='red',
        alpha=0.7, label='Fake',
        ax=axes[0,0]
    )
    axes[0,0].set_title('Tweet Count (Log)')
    axes[0,0].legend()

    # Plot 2: Title word count vs label
    df[df['real']==1]['title_word_count'].hist(
        bins=30, color='green',
        alpha=0.7, label='Real',
        ax=axes[0,1]
    )
    df[df['real']==0]['title_word_count'].hist(
        bins=30, color='red',
        alpha=0.7, label='Fake',
        ax=axes[0,1]
    )
    axes[0,1].set_title('Title Word Count')
    axes[0,1].legend()

    # Plot 3: URL length vs label
    df[df['real']==1]['url_length'].hist(
        bins=30, color='green',
        alpha=0.7, label='Real',
        ax=axes[0,2]
    )
    df[df['real']==0]['url_length'].hist(
        bins=30, color='red',
        alpha=0.7, label='Fake',
        ax=axes[0,2]
    )
    axes[0,2].set_title('URL Length')
    axes[0,2].legend()

    # Plot 4: Clickbait vs label
    clickbait_data = pd.crosstab(
        df['title_has_clickbait'],
        df['real']
    )
    clickbait_data.plot(
        kind='bar',
        ax=axes[1,0],
        color=['red','green'],
        edgecolor='black'
    )
    axes[1,0].set_title('Clickbait Words vs Label')
    axes[1,0].set_xticklabels(
        ['No Clickbait', 'Has Clickbait'],
        rotation=0
    )
    axes[1,0].legend(['Fake','Real'])

    # Plot 5: Domain trusted vs label
    trusted_data = pd.crosstab(
        df['domain_trusted'],
        df['real']
    )
    trusted_data.plot(
        kind='bar',
        ax=axes[1,1],
        color=['red','green'],
        edgecolor='black'
    )
    axes[1,1].set_title('Trusted Domain vs Label')
    axes[1,1].set_xticklabels(
        ['Not Trusted', 'Trusted'],
        rotation=0
    )
    axes[1,1].legend(['Fake','Real'])

    # Plot 6: HTTPS vs label
    https_data = pd.crosstab(
        df['url_has_https'],
        df['real']
    )
    https_data.plot(
        kind='bar',
        ax=axes[1,2],
        color=['red','green'],
        edgecolor='black'
    )
    axes[1,2].set_title('HTTPS vs Label')
    axes[1,2].set_xticklabels(
        ['HTTP', 'HTTPS'],
        rotation=0
    )
    axes[1,2].legend(['Fake','Real'])

    plt.tight_layout()
    plt.savefig('static/feature_analysis.png')
    plt.show()
    print("Saved to static/feature_analysis.png")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("="*60)
    print("FAKE NEWS DETECTION - PREPROCESSING")
    print("="*60)

    # Load data
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = os.path.join(BASE_DIR, 'data', 'dataset.csv')
    df = pd.read_csv(FILE_PATH)
    print(f"Dataset loaded: {len(df)} samples")

    # Step 1: Clean dataset
    df = clean_dataset(df)

    # Step 2: Clean title text
    print("\nSTEP 2: Cleaning Title Text...")
    print("="*50)
    print("This may take a moment...")
    df['cleaned_title'] = df['title'].apply(clean_text)
    print("✅ Title text cleaned!")
    print("\nSample:")
    print(f"Original : {df['title'][0]}")
    print(f"Cleaned  : {df['cleaned_title'][0]}")

    # Step 3: URL features
    df = extract_url_features(df)

    # Step 4: Domain features
    df = extract_domain_features(df)

    # Step 5: Title features
    df = extract_title_features(df)

    # Step 6: Tweet features
    df = extract_tweet_features(df)

    # Step 7: Handle imbalance
    df_balanced = handle_imbalance(df)

    
    # Step 8: Word cloud (disabled for speed)
# generate_wordcloud(df_balanced)
    print("Word cloud skipped for speed...")

    # Save preprocessed data
    df_balanced.to_csv(
        'data/preprocessed_data.csv',
        index=False
    )
    print("\n")
    print("="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"Final Dataset Size : {len(df_balanced)}")
    print(f"Real News          : {len(df_balanced[df_balanced['real']==1])}")
    print(f"Fake News          : {len(df_balanced[df_balanced['real']==0])}")
    print("\nFiles Saved:")
    print("- data/preprocessed_data.csv")
    print("- static/wordcloud.png")
    print("- static/feature_analysis.png")