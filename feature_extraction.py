import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import pickle
import os

# ============================================
# CREATE DIRECTORIES
# ============================================

os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ============================================
# STEP 1: LOAD PREPROCESSED DATA
# ============================================

def load_data():
    print("Loading Preprocessed Data...")
    print("="*50)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = os.path.join(
        BASE_DIR, 'data', 'preprocessed_data.csv'
    )

    df = pd.read_csv(FILE_PATH)

    # Fill any missing cleaned_title
    df['cleaned_title'] = df['cleaned_title'].fillna('')

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nLabel Distribution:")
    print(df['real'].value_counts())

    return df


# ============================================
# STEP 2: TF-IDF VECTORIZATION
# ============================================

def apply_tfidf(X_train, X_test):
    print("\nSTEP 2: Applying TF-IDF Vectorization...")
    print("="*50)

    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{2,}'
    )

    print("Fitting TF-IDF on training data...")
    X_train_tfidf = tfidf.fit_transform(X_train)

    print("Transforming test data...")
    X_test_tfidf = tfidf.transform(X_test)

    print(f"\nTF-IDF Shape:")
    print(f"  Training : {X_train_tfidf.shape}")
    print(f"  Test     : {X_test_tfidf.shape}")
    print(f"  Vocabulary Size: {len(tfidf.vocabulary_)}")

    # Save vectorizer
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("\nVectorizer saved to models/tfidf_vectorizer.pkl")

    return X_train_tfidf, X_test_tfidf, tfidf


# ============================================
# STEP 3: GET HANDCRAFTED FEATURES
# ============================================

def get_handcrafted_features(df):
    print("\nSTEP 3: Collecting Handcrafted Features...")
    print("="*50)

    # All numeric features extracted in preprocessing
    feature_columns = [
        # URL features (8)
        'url_length',
        'url_dots',
        'url_slashes',
        'url_has_https',
        'url_path_length',
        'url_has_numbers',
        'url_has_special',
        'url_hyphens',

        # Domain features (5)
        'domain_length',
        'domain_trusted',
        'domain_numbers',
        'domain_has_www',
        'domain_extension',

        # Title features (9)
        'title_word_count',
        'title_char_count',
        'title_has_numbers',
        'title_uppercase_count',
        'title_exclamation',
        'title_question',
        'title_avg_word_length',
        'title_has_clickbait',
        'title_clickbait_count',

        # Tweet features (4)
        'tweet_num',
        'tweet_log',
        'tweet_category',
        'is_viral'
    ]

    # Check which columns exist
    available_cols = [
        col for col in feature_columns
        if col in df.columns
    ]

    missing_cols = [
        col for col in feature_columns
        if col not in df.columns
    ]

    print(f"Available features : {len(available_cols)}")
    print(f"Missing features   : {len(missing_cols)}")

    if missing_cols:
        print(f"Missing columns: {missing_cols}")

    # Extract features
    features = df[available_cols].copy()

    # Fill any missing values
    features = features.fillna(0)

    print(f"\nHandcrafted Features Shape: {features.shape}")
    print(f"Feature Names:")
    for i, col in enumerate(available_cols, 1):
        print(f"  {i:2d}. {col}")

    return features, available_cols


# ============================================
# STEP 4: COMBINE ALL FEATURES
# ============================================

def combine_features(X_tfidf, handcrafted_features):
    print("\nSTEP 4: Combining TF-IDF + Handcrafted Features...")
    print("="*50)

    # Convert handcrafted to sparse matrix
    handcrafted_sparse = csr_matrix(
        handcrafted_features.values
    )

    # Combine TF-IDF + Handcrafted
    X_combined = hstack([X_tfidf, handcrafted_sparse])

    print(f"TF-IDF Shape       : {X_tfidf.shape}")
    print(f"Handcrafted Shape  : {handcrafted_sparse.shape}")
    print(f"Combined Shape     : {X_combined.shape}")

    return X_combined


# ============================================
# STEP 5: SPLIT DATA
# ============================================

def split_data(df):
    print("\nSTEP 5: Splitting Data...")
    print("="*50)

    X_text = df['cleaned_title']
    y = df['real']

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training Set : {len(X_train)} samples")
    print(f"Test Set     : {len(X_test)} samples")
    print(f"\nTraining Label Distribution:")
    print(y_train.value_counts())
    print(f"\nTest Label Distribution:")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test


# ============================================
# STEP 6: FEATURE IMPORTANCE ANALYSIS
# ============================================

def feature_importance_analysis(df):
    print("\nSTEP 6: Feature Importance Analysis...")
    print("="*50)

    feature_columns = [
        'url_length', 'url_dots', 'url_slashes',
        'url_has_https', 'url_path_length',
        'url_has_numbers', 'url_has_special',
        'url_hyphens', 'domain_length',
        'domain_trusted', 'domain_numbers',
        'domain_has_www', 'domain_extension',
        'title_word_count', 'title_char_count',
        'title_has_numbers', 'title_uppercase_count',
        'title_exclamation', 'title_question',
        'title_avg_word_length', 'title_has_clickbait',
        'title_clickbait_count', 'tweet_num',
        'tweet_log', 'tweet_category', 'is_viral'
    ]

    available = [
        col for col in feature_columns
        if col in df.columns
    ]

    # Calculate correlation with label
    correlations = []
    for col in available:
        corr = df[col].corr(df['real'])
        correlations.append({
            'feature': col,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values(
        'abs_correlation',
        ascending=False
    )

    print("\nFeature Correlation with Label (real):")
    print("-"*50)
    for _, row in corr_df.iterrows():
        direction = "↑ Real" if row['correlation'] > 0 else "↑ Fake"
        print(f"  {row['feature']:<30} "
              f"{row['correlation']:>8.4f}  {direction}")

    # Plot top features
    plt.figure(figsize=(12, 8))
    colors = [
        'green' if c > 0 else 'red'
        for c in corr_df['correlation']
    ]
    plt.barh(
        corr_df['feature'],
        corr_df['correlation'],
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    plt.axvline(x=0, color='black', linewidth=1)
    plt.title(
        'Feature Correlation with Label\n'
        '(Green = Real, Red = Fake)',
        fontsize=14,
        fontweight='bold'
    )
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.show()
    print("\nSaved to static/feature_importance.png")

    return corr_df


# ============================================
# STEP 7: VISUALIZE FEATURE DISTRIBUTIONS
# ============================================

def visualize_distributions(df):
    print("\nSTEP 7: Visualizing Feature Distributions...")

    key_features = [
        'tweet_log',
        'title_word_count',
        'url_length',
        'domain_trusted',
        'title_has_clickbait',
        'title_exclamation'
    ]

    available = [
        f for f in key_features
        if f in df.columns
    ]

    n = len(available)
    fig, axes = plt.subplots(
        2, 3,
        figsize=(18, 10)
    )
    fig.suptitle(
        'Key Feature Distributions (Real vs Fake)',
        fontsize=16,
        fontweight='bold'
    )

    axes = axes.flatten()

    for i, feature in enumerate(available):
        if i < len(axes):
            df[df['real']==1][feature].hist(
                bins=30, color='green',
                alpha=0.7, label='Real',
                ax=axes[i]
            )
            df[df['real']==0][feature].hist(
                bins=30, color='red',
                alpha=0.7, label='Fake',
                ax=axes[i]
            )
            axes[i].set_title(
                feature.replace('_', ' ').title(),
                fontsize=11
            )
            axes[i].legend()

    # Hide unused subplots
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig('static/feature_distributions.png')
    plt.show()
    print("Saved to static/feature_distributions.png")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("="*60)
    print("FAKE NEWS DETECTION - FEATURE EXTRACTION")
    print("="*60)

    # Step 1: Load data
    df = load_data()

    # Step 2: Split data
    X_train_text, X_test_text, y_train, y_test = split_data(df)

    # Step 3: TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf = apply_tfidf(
        X_train_text, X_test_text
    )

    # Step 4: Handcrafted features
    handcrafted_features, feature_names = get_handcrafted_features(df)

    # Split handcrafted features same way
    from sklearn.model_selection import train_test_split as tts
    hf_train, hf_test = tts(
        handcrafted_features,
        test_size=0.2,
        random_state=42,
        stratify=df['real']
    )

    # Step 5: Combine features
    X_train_combined = combine_features(
        X_train_tfidf, hf_train
    )
    X_test_combined = combine_features(
        X_test_tfidf, hf_test
    )

    # Step 6: Feature importance
    corr_df = feature_importance_analysis(df)

    # Step 7: Visualize
    visualize_distributions(df)

    # Save feature names
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    # Save combined features info
    feature_info = {
        'tfidf_features': X_train_tfidf.shape[1],
        'handcrafted_features': len(feature_names),
        'total_features': X_train_combined.shape[1],
        'feature_names': feature_names,
        'training_samples': X_train_combined.shape[0],
        'test_samples': X_test_combined.shape[0]
    }

    with open('models/feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)

    print("\n")
    print("="*60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nFeature Summary:")
    print(f"  TF-IDF Features      : {X_train_tfidf.shape[1]}")
    print(f"  Handcrafted Features : {len(feature_names)}")
    print(f"  Total Features       : {X_train_combined.shape[1]}")
    print(f"  Training Samples     : {X_train_combined.shape[0]}")
    print(f"  Test Samples         : {X_test_combined.shape[0]}")
    print(f"\nFiles Saved:")
    print(f"  - models/tfidf_vectorizer.pkl")
    print(f"  - models/feature_names.pkl")
    print(f"  - models/feature_info.pkl")
    print(f"  - static/feature_importance.png")
    print(f"  - static/feature_distributions.png")