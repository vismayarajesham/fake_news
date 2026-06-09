import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
# CREATE DIRECTORIES
# ============================================

os.makedirs('data', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ============================================
# LOAD YOUR DATASET
# ============================================

def load_data():

    # ----------------------------------------
    # Load your CSV file
    # Change the filename to your file name
    # ----------------------------------------
    df = pd.read_csv('data/dataset.csv')

    print("Dataset Loaded Successfully!")
    print("="*50)
    print(f"Total Samples     : {len(df)}")
    print(f"\nColumn Names:")
    print(df.columns.tolist())
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nFirst 5 Rows:")
    print(df.head())

    return df


# ============================================
# VALIDATE DATASET
# ============================================

def validate_data(df):
    print("\n")
    print("="*50)
    print("DATASET VALIDATION")
    print("="*50)

    # Check required columns
    required_columns = [
        'title',
        'news_url',
        'source_domain',
        'tweet_num',
        'real'
    ]

    print("\nChecking Required Columns...")
    for col in required_columns:
        if col in df.columns:
            print(f"  ✅ '{col}' column found")
        else:
            print(f"  ❌ '{col}' column MISSING")

    # Check label values
    print(f"\nLabel Values Found:")
    print(df['real'].unique())
    print(f"\nLabel Distribution:")
    print(df['real'].value_counts())

    # Check missing values
    print(f"\nMissing Values:")
    print(df.isnull().sum())

    # Check duplicates
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")

    return df


# ============================================
# EXPLORE DATASET
# ============================================

def explore_data(df):
    print("\n")
    print("="*50)
    print("DATASET EXPLORATION")
    print("="*50)

    # Total samples
    total     = len(df)
    real_news = len(df[df['real'] == 1])
    fake_news = len(df[df['real'] == 0])

    print(f"\nTotal Samples  : {total}")
    print(f"Real News (1)  : {real_news} "
          f"({real_news/total*100:.2f}%)")
    print(f"Fake News (0)  : {fake_news} "
          f"({fake_news/total*100:.2f}%)")

    # Title statistics
    print(f"\nTitle Statistics:")
    df['title_length'] = df['title'].apply(
        lambda x: len(str(x).split())
    )
    print(df['title_length'].describe())

    # Tweet number statistics
    print(f"\nTweet Number Statistics:")
    print(df['tweet_num'].describe())

    # Source domain
    print(f"\nTop 10 Source Domains:")
    print(df['source_domain'].value_counts().head(10))

    # Real vs Fake by source domain
    print(f"\nReal vs Fake by Top 5 Domains:")
    top_domains = df['source_domain'].value_counts(
    ).head(5).index
    for domain in top_domains:
        domain_df = df[df['source_domain'] == domain]
        real = len(domain_df[domain_df['real'] == 1])
        fake = len(domain_df[domain_df['real'] == 0])
        print(f"  {domain}: Real={real}, Fake={fake}")

    return df


# ============================================
# VISUALIZE DATASET
# ============================================

def visualize_data(df):
    print("\nGenerating Visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        'Dataset Analysis',
        fontsize=16,
        fontweight='bold'
    )

    # ----------------------------------------
    # Plot 1: Class Distribution
    # ----------------------------------------
    colors = ['#FF6B6B', '#51CF66']
    labels = ['Fake News (0)', 'Real News (1)']
    values = [
        len(df[df['real'] == 0]),
        len(df[df['real'] == 1])
    ]

    axes[0, 0].bar(
        labels, values,
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    axes[0, 0].set_title(
        'Class Distribution',
        fontsize=13,
        fontweight='bold'
    )
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(values):
        axes[0, 0].text(
            i, v + 50,
            str(v),
            ha='center',
            fontweight='bold'
        )

    # ----------------------------------------
    # Plot 2: Tweet Number Distribution
    # ----------------------------------------
    df[df['real'] == 1]['tweet_num'].hist(
        bins=50,
        color='green',
        alpha=0.7,
        label='Real News',
        ax=axes[0, 1]
    )
    df[df['real'] == 0]['tweet_num'].hist(
        bins=50,
        color='red',
        alpha=0.7,
        label='Fake News',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title(
        'Tweet Count Distribution',
        fontsize=13,
        fontweight='bold'
    )
    axes[0, 1].set_xlabel('Number of Tweets')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()

    # ----------------------------------------
    # Plot 3: Title Length Distribution
    # ----------------------------------------
    df['title_length'] = df['title'].apply(
        lambda x: len(str(x).split())
    )
    df[df['real'] == 1]['title_length'].hist(
        bins=30,
        color='green',
        alpha=0.7,
        label='Real News',
        ax=axes[1, 0]
    )
    df[df['real'] == 0]['title_length'].hist(
        bins=30,
        color='red',
        alpha=0.7,
        label='Fake News',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title(
        'Title Length Distribution',
        fontsize=13,
        fontweight='bold'
    )
    axes[1, 0].set_xlabel('Title Word Count')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()

    # ----------------------------------------
    # Plot 4: Top 10 Source Domains
    # ----------------------------------------
    top_domains = df['source_domain'].value_counts(
    ).head(10)
    axes[1, 1].barh(
        top_domains.index,
        top_domains.values,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )
    axes[1, 1].set_title(
        'Top 10 Source Domains',
        fontsize=13,
        fontweight='bold'
    )
    axes[1, 1].set_xlabel('Count')
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('static/dataset_analysis.png')
    plt.show()
    print("Saved to static/dataset_analysis.png")


# ============================================
# CLEAN AND PREPARE
# ============================================

def prepare_data(df):
    print("\n")
    print("="*50)
    print("PREPARING DATASET")
    print("="*50)

    # Step 1: Remove duplicates
    print(f"Before removing duplicates: {len(df)}")
    df = df.drop_duplicates()
    print(f"After removing duplicates : {len(df)}")

    # Step 2: Drop missing values
    print(f"Before dropping NA: {len(df)}")
    df = df.dropna(subset=['title', 'real'])
    print(f"After dropping NA : {len(df)}")

    # Step 3: Reset index
    df = df.reset_index(drop=True)

    # Step 4: Ensure label is integer
    df['real'] = df['real'].astype(int)

    # Step 5: Clean title
    df['title'] = df['title'].astype(str)
    df['source_domain'] = df['source_domain'].astype(str)

    # Step 6: Fill missing tweet_num with 0
    df['tweet_num'] = df['tweet_num'].fillna(0)

    print(f"\nFinal Dataset Size: {len(df)}")
    print(f"\nFinal Label Distribution:")
    print(df['real'].value_counts())
    print(f"\nSample Data:")
    print(df[['title', 'source_domain',
              'tweet_num', 'real']].head())

    return df


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("="*60)
    print("FAKE NEWS DETECTION - DATA LOADING")
    print("="*60)

    # Load data
    df = load_data()

    # Validate data
    df = validate_data(df)

    # Explore data
    df = explore_data(df)

    # Visualize data
    visualize_data(df)

    # Prepare data
    df = prepare_data(df)

    # Save prepared data
    df.to_csv('data/combined_data.csv', index=False)
    print("\n")
    print("="*60)
    print("DATA LOADING COMPLETE!")
    print("="*60)
    print("Files Saved:")
    print("- data/combined_data.csv")