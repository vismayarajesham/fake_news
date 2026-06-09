import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    train_test_split
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix

# ============================================
# CREATE DIRECTORIES
# ============================================

os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)

# ============================================
# STEP 1: LOAD DATA
# ============================================

def load_all_data():
    print("Loading All Data...")
    print("="*50)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load preprocessed data
    df = pd.read_csv(
        os.path.join(BASE_DIR, 'data', 'preprocessed_data.csv')
    )
    df['cleaned_title'] = df['cleaned_title'].fillna('')

    # Load TF-IDF vectorizer
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    # Load feature names
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    print(f"Dataset    : {len(df)} samples")
    print(f"Features   : {len(feature_names)} handcrafted")
    print(f"Label Dist :")
    print(df['real'].value_counts())

    return df, tfidf, feature_names


# ============================================
# STEP 2: PREPARE FEATURES
# ============================================

def prepare_features(df, tfidf, feature_names):
    print("\nPreparing Features...")
    print("="*50)

    # Text features
    X_text = df['cleaned_title']
    y = df['real']

    # Split data
    (X_train_text, X_test_text,
     y_train, y_test) = train_test_split(
        X_text, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF transform
    X_train_tfidf = tfidf.transform(X_train_text)
    X_test_tfidf  = tfidf.transform(X_test_text)

    # Handcrafted features
    handcrafted = df[feature_names].fillna(0)

    # Scale handcrafted features
    scaler = MinMaxScaler()
    handcrafted_scaled = scaler.fit_transform(handcrafted)

    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Split handcrafted
    (hf_train, hf_test,
     _, _) = train_test_split(
        handcrafted_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Combine TF-IDF + Handcrafted
    X_train = hstack([
        X_train_tfidf,
        csr_matrix(hf_train)
    ])
    X_test = hstack([
        X_test_tfidf,
        csr_matrix(hf_test)
    ])

    print(f"Training Features Shape : {X_train.shape}")
    print(f"Test Features Shape     : {X_test.shape}")
    print(f"Training Labels         : {len(y_train)}")
    print(f"Test Labels             : {len(y_test)}")

    return X_train, X_test, y_train, y_test


# ============================================
# STEP 3: DEFINE MODELS
# ============================================

def get_models():
    print("\nDefining Models...")
    print("="*50)

    models = {

        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            class_weight='balanced',
            solver='lbfgs'
        ),

        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=20,
            class_weight='balanced'
        ),

        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=20,
            class_weight='balanced'
        ),

        'Linear SVC': LinearSVC(
            random_state=42,
            max_iter=2000,
            C=1.0,
            class_weight='balanced'
        ),

        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,
            learning_rate=0.1
        )
    }

    print(f"Models defined: {list(models.keys())}")
    return models


# ============================================
# STEP 4: TRAIN AND EVALUATE EACH MODEL
# ============================================

def train_and_evaluate(model, model_name,
                        X_train, X_test,
                        y_train, y_test):
    print(f"\nTraining {model_name}...")
    print("-"*40)

    # Train
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    training_time = round(end - start, 2)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    report    = classification_report(
        y_test, y_pred,
        target_names=['Fake News', 'Real News'],
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Training Time : {training_time}s")
    print(f"Accuracy      : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Fake News', 'Real News']
    ))

    return model, y_pred, accuracy, report, cm, training_time


# ============================================
# STEP 5: CROSS VALIDATION
# ============================================

def cross_validate(model, X_train, y_train, model_name):
    print(f"Cross Validating {model_name}...")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1
    )

    print(f"CV Scores : {np.round(cv_scores, 4)}")
    print(f"CV Mean   : {cv_scores.mean():.4f}")
    print(f"CV Std    : {cv_scores.std():.4f}")

    return cv_scores


# ============================================
# STEP 6: PLOT CONFUSION MATRIX
# ============================================

def plot_confusion_matrix(cm, model_name, accuracy):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Fake', 'Real'],
        yticklabels=['Fake', 'Real'],
        ax=axes[0],
        linewidths=0.5
    )
    axes[0].set_title(
        f'{model_name}\nConfusion Matrix (Raw)',
        fontweight='bold'
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')

    # Normalized
    cm_norm = cm.astype('float') / cm.sum(
        axis=1
    )[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.3f',
        cmap='Greens',
        xticklabels=['Fake', 'Real'],
        yticklabels=['Fake', 'Real'],
        ax=axes[1],
        linewidths=0.5
    )
    axes[1].set_title(
        f'{model_name}\nConfusion Matrix (Normalized)',
        fontweight='bold'
    )
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')

    plt.suptitle(
        f'Accuracy: {accuracy*100:.2f}%',
        fontsize=12
    )
    plt.tight_layout()

    filename = model_name.lower().replace(' ', '_')
    plt.savefig(f'static/cm_{filename}.png')
    plt.show()
    print(f"Saved to static/cm_{filename}.png")


# ============================================
# STEP 7: COMPARE ALL MODELS
# ============================================

def compare_all_models(all_results):
    print("\n")
    print("="*60)
    print("ALL MODELS COMPARISON")
    print("="*60)

    # Create dataframe
    results_data = []
    for name, res in all_results.items():
        results_data.append({
            'Model'         : name,
            'Accuracy'      : res['accuracy'],
            'CV Mean'       : res['cv_mean'],
            'CV Std'        : res['cv_std'],
            'Training Time' : res['training_time'],
            'F1 Fake'       : res['report']['Fake News']['f1-score'],
            'F1 Real'       : res['report']['Real News']['f1-score']
        })

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(
        'Accuracy',
        ascending=False
    ).reset_index(drop=True)

    print("\nRanked Results:")
    print("-"*60)
    print(results_df.to_string(index=False))

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Model Comparison',
        fontsize=16,
        fontweight='bold'
    )

    colors = plt.cm.viridis(
        np.linspace(0, 1, len(results_df))
    )

    # Accuracy
    bars1 = axes[0].bar(
        results_df['Model'],
        results_df['Accuracy'],
        color=colors,
        edgecolor='black'
    )
    axes[0].set_title('Accuracy')
    axes[0].set_ylim(0.5, 1.0)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, results_df['Accuracy']):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=8
        )

    # CV Mean
    bars2 = axes[1].bar(
        results_df['Model'],
        results_df['CV Mean'],
        color=colors,
        edgecolor='black'
    )
    axes[1].set_title('Cross Validation Mean')
    axes[1].set_ylim(0.5, 1.0)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, results_df['CV Mean']):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=8
        )

    # Training Time
    bars3 = axes[2].bar(
        results_df['Model'],
        results_df['Training Time'],
        color=colors,
        edgecolor='black'
    )
    axes[2].set_title('Training Time (seconds)')
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, results_df['Training Time']):
        axes[2].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.1,
            f'{val:.1f}s',
            ha='center', va='bottom',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig('static/model_comparison.png')
    plt.show()
    print("\nSaved to static/model_comparison.png")

    return results_df


# ============================================
# STEP 8: SAVE BEST MODEL
# ============================================

def save_best_model(trained_models, results_df):
    print("\nSaving Best Model...")
    print("="*50)

    # Get best model name
    best_name = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Accuracy']
    best_model = trained_models[best_name]

    print(f"Best Model    : {best_name}")
    print(f"Best Accuracy : {best_accuracy:.4f} "
          f"({best_accuracy*100:.2f}%)")

    # Save best model
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Save model info
    model_info = {
        'model_name' : best_name,
        'accuracy'   : best_accuracy
    }
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)

    print("\nFiles Saved:")
    print("  - models/best_model.pkl")
    print("  - models/model_info.pkl")

    return best_model, best_name, best_accuracy


# ============================================
# STEP 9: ROC CURVES
# ============================================

def plot_roc_curves(trained_models, X_test, y_test):
    print("\nPlotting ROC Curves...")

    plt.figure(figsize=(10, 8))
    colors = [
        'blue', 'red', 'green',
        'orange', 'purple'
    ]

    for (name, model), color in zip(
        trained_models.items(), colors
    ):
        try:
            if hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(
                    X_test
                )[:, 1]
            else:
                y_scores = model.decision_function(
                    X_test
                )
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            auc = roc_auc_score(y_test, y_scores)
            plt.plot(
                fpr, tpr,
                color=color,
                linewidth=2,
                label=f'{name} (AUC={auc:.4f})'
            )
        except Exception as e:
            print(f"ROC failed for {name}: {e}")

    plt.plot(
        [0,1],[0,1],
        'k--',
        label='Random (AUC=0.5)'
    )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(
        'ROC Curves - All Models',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/roc_curves.png')
    plt.show()
    print("Saved to static/roc_curves.png")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("="*60)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("="*60)

    # Step 1: Load data
    df, tfidf, feature_names = load_all_data()

    # Step 2: Prepare features
    X_train, X_test, y_train, y_test = prepare_features(
        df, tfidf, feature_names
    )

    # Step 3: Get models
    models = get_models()
    # 'Gradient Boosting': GradientBoostingClassifier(
#     n_estimators=100,
#     random_state=42,
#     max_depth=5,
#     learning_rate=0.1
# )

    # Store results
    all_results    = {}
    trained_models = {}

    # Step 4: Train all models
    print("\n")
    print("="*60)
    print("TRAINING ALL MODELS")
    print("="*60)

    for model_name, model in models.items():

        # Train and evaluate
        (trained_model,
         y_pred,
         accuracy,
         report,
         cm,
         training_time) = train_and_evaluate(
            model, model_name,
            X_train, X_test,
            y_train, y_test
        )

        # Cross validation
        cv_scores = cross_validate(
            trained_model,
            X_train, y_train,
            model_name
        )

        # Plot confusion matrix
        plot_confusion_matrix(cm, model_name, accuracy)

        # Store results
        all_results[model_name] = {
            'accuracy'      : accuracy,
            'cv_mean'       : cv_scores.mean(),
            'cv_std'        : cv_scores.std(),
            'training_time' : training_time,
            'report'        : report,
            'cm'            : cm
        }

        trained_models[model_name] = trained_model

        # Save individual model
        filename = model_name.lower().replace(' ', '_')
        with open(f'models/{filename}.pkl', 'wb') as f:
            pickle.dump(trained_model, f)
        print(f"Saved: models/{filename}.pkl")

    # Step 5: Compare models
    results_df = compare_all_models(all_results)

    # Step 6: ROC Curves
    plot_roc_curves(trained_models, X_test, y_test)

    # Step 7: Save best model
    best_model, best_name, best_acc = save_best_model(
        trained_models, results_df
    )

    # Final Summary
    print("\n")
    print("="*60)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*60)
    print(f"\nBest Model    : {best_name}")
    print(f"Best Accuracy : {best_acc:.4f} "
          f"({best_acc*100:.2f}%)")
    print(f"\nAll Models Ranked:")
    print(results_df[[
        'Model','Accuracy','CV Mean','Training Time'
    ]].to_string(index=False))
    print(f"\nFiles Saved:")
    print(f"  - models/best_model.pkl")
    print(f"  - models/model_info.pkl")
    print(f"  - models/scaler.pkl")
    print(f"  - models/logistic_regression.pkl")
    print(f"  - models/random_forest.pkl")
    print(f"  - models/decision_tree.pkl")
    print(f"  - models/linear_svc.pkl")
    print(f"  - models/gradient_boosting.pkl")
    print(f"  - static/model_comparison.png")
    print(f"  - static/roc_curves.png")