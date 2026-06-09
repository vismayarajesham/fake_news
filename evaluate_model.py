import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score
)
from sklearn.model_selection import (
    train_test_split,
    learning_curve
)
from scipy.sparse import hstack, csr_matrix

# ============================================
# CREATE DIRECTORIES
# ============================================

os.makedirs('static',  exist_ok=True)
os.makedirs('models',  exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ============================================
# STEP 1: LOAD EVERYTHING
# ============================================

def load_everything():
    print("Loading Model and Data...")
    print("="*50)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Load best model
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load model info
    with open('models/model_info.pkl', 'rb') as f:
        model_info = pickle.load(f)

    # Load TF-IDF vectorizer
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    # Load feature names
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)

    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load preprocessed data
    df = pd.read_csv(
        os.path.join(BASE_DIR, 'data', 'preprocessed_data.csv')
    )
    df['cleaned_title'] = df['cleaned_title'].fillna('')

    print(f"✅ Model      : {model_info['model_name']}")
    print(f"✅ Accuracy   : {model_info['accuracy']:.4f}")
    print(f"✅ Dataset    : {len(df)} samples")
    print(f"✅ Features   : {len(feature_names)} handcrafted")

    return model, model_info, tfidf, feature_names, scaler, df


# ============================================
# STEP 2: PREPARE TEST DATA
# ============================================

def prepare_test_data(df, tfidf, feature_names, scaler):
    print("\nPreparing Test Data...")
    print("="*50)

    X_text = df['cleaned_title']
    y = df['real']

    # Split same way as training
    (X_train_text, X_test_text,
     y_train, y_test) = train_test_split(
        X_text, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF
    X_train_tfidf = tfidf.transform(X_train_text)
    X_test_tfidf  = tfidf.transform(X_test_text)

    # Handcrafted features
    handcrafted = df[feature_names].fillna(0)
    handcrafted_scaled = scaler.transform(handcrafted)

    # Split handcrafted
    (hf_train, hf_test,
     _, _) = train_test_split(
        handcrafted_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Combine
    X_train = hstack([
        X_train_tfidf,
        csr_matrix(hf_train)
    ])
    X_test = hstack([
        X_test_tfidf,
        csr_matrix(hf_test)
    ])

    print(f"Test Set Shape : {X_test.shape}")
    print(f"Test Labels    : {len(y_test)}")
    print(f"Real News      : {sum(y_test==1)}")
    print(f"Fake News      : {sum(y_test==0)}")

    return X_train, X_test, y_train, y_test


# ============================================
# STEP 3: COMPREHENSIVE METRICS
# ============================================

def comprehensive_metrics(y_test, y_pred,
                           y_scores=None):
    print("\n")
    print("="*60)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("="*60)

    # Basic metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average='weighted'
    )
    recall    = recall_score(
        y_test, y_pred, average='weighted'
    )
    f1        = f1_score(
        y_test, y_pred, average='weighted'
    )
    mcc       = matthews_corrcoef(y_test, y_pred)
    kappa     = cohen_kappa_score(y_test, y_pred)

    # Per class
    prec_fake = precision_score(
        y_test, y_pred, pos_label=0
    )
    prec_real = precision_score(
        y_test, y_pred, pos_label=1
    )
    rec_fake  = recall_score(
        y_test, y_pred, pos_label=0
    )
    rec_real  = recall_score(
        y_test, y_pred, pos_label=1
    )
    f1_fake   = f1_score(
        y_test, y_pred, pos_label=0
    )
    f1_real   = f1_score(
        y_test, y_pred, pos_label=1
    )

    print(f"\n{'Metric':<30} {'Score':>10}")
    print("-"*42)
    print(f"{'Accuracy':<30} {accuracy:>10.4f}")
    print(f"{'Precision (Weighted)':<30} {precision:>10.4f}")
    print(f"{'Recall (Weighted)':<30} {recall:>10.4f}")
    print(f"{'F1-Score (Weighted)':<30} {f1:>10.4f}")
    print(f"{'Matthews Corr Coef':<30} {mcc:>10.4f}")
    print(f"{'Cohen Kappa Score':<30} {kappa:>10.4f}")

    # AUC-ROC
    auc = None
    ap  = None
    if y_scores is not None:
        auc = roc_auc_score(y_test, y_scores)
        ap  = average_precision_score(
            y_test, y_scores
        )
        print(f"{'AUC-ROC Score':<30} {auc:>10.4f}")
        print(f"{'Average Precision':<30} {ap:>10.4f}")

    print(f"\n{'Per Class Metrics':}")
    print("-"*55)
    print(f"{'Class':<20} {'Precision':>10} "
          f"{'Recall':>10} {'F1':>10}")
    print("-"*55)
    print(f"{'Fake News (0)':<20} {prec_fake:>10.4f} "
          f"{rec_fake:>10.4f} {f1_fake:>10.4f}")
    print(f"{'Real News (1)':<20} {prec_real:>10.4f} "
          f"{rec_real:>10.4f} {f1_real:>10.4f}")

    metrics = {
        'Accuracy'         : accuracy,
        'Precision'        : precision,
        'Recall'           : recall,
        'F1-Score'         : f1,
        'MCC'              : mcc,
        'Kappa'            : kappa,
        'AUC-ROC'          : auc,
        'Avg Precision'    : ap,
        'Precision (Fake)' : prec_fake,
        'Precision (Real)' : prec_real,
        'Recall (Fake)'    : rec_fake,
        'Recall (Real)'    : rec_real,
        'F1 (Fake)'        : f1_fake,
        'F1 (Real)'        : f1_real
    }

    return metrics


# ============================================
# STEP 4: DETAILED CONFUSION MATRIX
# ============================================

def detailed_confusion_matrix(y_test, y_pred):
    print("\nGenerating Detailed Confusion Matrix...")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\nConfusion Matrix Values:")
    print(f"  True Negative  (TN) : {tn}")
    print(f"  False Positive (FP) : {fp}")
    print(f"  False Negative (FN) : {fn}")
    print(f"  True Positive  (TP) : {tp}")

    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    fpr_val     = fp/(fp+tn)
    fnr_val     = fn/(fn+tp)
    ppv         = tp/(tp+fp)
    npv         = tn/(tn+fn)

    print(f"\nDerived Metrics:")
    print(f"  Sensitivity (TPR) : {sensitivity:.4f}")
    print(f"  Specificity (TNR) : {specificity:.4f}")
    print(f"  False Pos Rate    : {fpr_val:.4f}")
    print(f"  False Neg Rate    : {fnr_val:.4f}")
    print(f"  Pos Pred Value    : {ppv:.4f}")
    print(f"  Neg Pred Value    : {npv:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Fake', 'Real'],
        yticklabels=['Fake', 'Real'],
        ax=axes[0],
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    axes[0].set_title(
        'Confusion Matrix (Raw Counts)',
        fontsize=13,
        fontweight='bold'
    )
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

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
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    axes[1].set_title(
        'Confusion Matrix (Normalized)',
        fontsize=13,
        fontweight='bold'
    )
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    plt.suptitle(
        f'Gradient Boosting - Confusion Matrix Analysis',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        'static/detailed_confusion_matrix.png',
        bbox_inches='tight'
    )
    plt.show()
    print("Saved to static/detailed_confusion_matrix.png")

    return cm, tn, fp, fn, tp


# ============================================
# STEP 5: ROC CURVE
# ============================================

def roc_curve_analysis(y_test, y_scores):
    print("\nGenerating ROC Curve...")

    fpr, tpr, thresholds = roc_curve(
        y_test, y_scores
    )
    auc = roc_auc_score(y_test, y_scores)

    # Optimal threshold
    optimal_idx       = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr       = tpr[optimal_idx]
    optimal_fpr       = fpr[optimal_idx]

    print(f"\nROC Analysis:")
    print(f"  AUC Score         : {auc:.4f}")
    print(f"  Optimal Threshold : {optimal_threshold:.4f}")
    print(f"  Optimal TPR       : {optimal_tpr:.4f}")
    print(f"  Optimal FPR       : {optimal_fpr:.4f}")

    plt.figure(figsize=(9, 7))
    plt.plot(
        fpr, tpr,
        color='darkorange',
        linewidth=2.5,
        label=f'ROC Curve (AUC = {auc:.4f})'
    )
    plt.scatter(
        optimal_fpr, optimal_tpr,
        color='red', s=120, zorder=5,
        label=f'Optimal Point\n'
              f'Threshold={optimal_threshold:.4f}'
    )
    plt.plot(
        [0,1],[0,1],
        'k--', linewidth=1,
        label='Random (AUC=0.5)'
    )
    plt.fill_between(
        fpr, tpr,
        alpha=0.1, color='darkorange'
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(
        'ROC Curve - Fake News Detection\n'
        f'(Best Model: Gradient Boosting)',
        fontsize=13,
        fontweight='bold'
    )
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/roc_curve_analysis.png')
    plt.show()
    print("Saved to static/roc_curve_analysis.png")

    return auc, optimal_threshold


# ============================================
# STEP 6: PRECISION RECALL CURVE
# ============================================

def precision_recall_analysis(y_test, y_scores):
    print("\nGenerating Precision-Recall Curve...")

    precision, recall, thresholds = precision_recall_curve(
        y_test, y_scores
    )
    ap = average_precision_score(y_test, y_scores)

    # Best F1
    f1_scores = 2*(precision*recall)/(
        precision+recall+1e-8
    )
    best_idx = np.argmax(f1_scores)
    best_f1  = f1_scores[best_idx]
    best_thr = thresholds[best_idx] \
        if best_idx < len(thresholds) \
        else thresholds[-1]

    print(f"\nPR Analysis:")
    print(f"  Average Precision : {ap:.4f}")
    print(f"  Best F1-Score     : {best_f1:.4f}")
    print(f"  Best Threshold    : {best_thr:.4f}")

    plt.figure(figsize=(9, 7))
    plt.plot(
        recall, precision,
        color='blue', linewidth=2.5,
        label=f'PR Curve (AP={ap:.4f})'
    )
    plt.scatter(
        recall[best_idx],
        precision[best_idx],
        color='red', s=120, zorder=5,
        label=f'Best F1={best_f1:.4f}'
    )
    baseline = len(y_test[y_test==1])/len(y_test)
    plt.axhline(
        y=baseline, color='gray',
        linestyle='--',
        label=f'Baseline={baseline:.4f}'
    )
    plt.fill_between(
        recall, precision,
        alpha=0.1, color='blue'
    )
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(
        'Precision-Recall Curve\n'
        '(Best Model: Gradient Boosting)',
        fontsize=13,
        fontweight='bold'
    )
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/precision_recall_curve.png')
    plt.show()
    print("Saved to static/precision_recall_curve.png")

    return ap, best_thr


# ============================================
# STEP 7: METRICS VISUALIZATION
# ============================================

def visualize_metrics(metrics):
    print("\nVisualizing Metrics...")

    main_metrics = {
        'Accuracy'  : metrics['Accuracy'],
        'Precision' : metrics['Precision'],
        'Recall'    : metrics['Recall'],
        'F1-Score'  : metrics['F1-Score'],
        'AUC-ROC'   : metrics['AUC-ROC'] \
            if metrics['AUC-ROC'] else 0,
        'MCC'       : metrics['MCC']
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Model Evaluation Metrics Summary',
        fontsize=15,
        fontweight='bold'
    )

    # Bar chart
    colors = [
        '#2196F3','#4CAF50','#FF9800',
        '#E91E63','#9C27B0','#00BCD4'
    ]
    bars = axes[0].bar(
        main_metrics.keys(),
        main_metrics.values(),
        color=colors,
        edgecolor='black',
        linewidth=0.5
    )
    axes[0].set_title(
        'Overall Metrics',
        fontsize=13,
        fontweight='bold'
    )
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1.1)
    axes[0].tick_params(axis='x', rotation=30)
    for bar, val in zip(
        bars, main_metrics.values()
    ):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )

    # Per class metrics
    categories = ['Precision','Recall','F1-Score']
    fake_vals  = [
        metrics['Precision (Fake)'],
        metrics['Recall (Fake)'],
        metrics['F1 (Fake)']
    ]
    real_vals  = [
        metrics['Precision (Real)'],
        metrics['Recall (Real)'],
        metrics['F1 (Real)']
    ]

    x     = np.arange(len(categories))
    width = 0.35

    b1 = axes[1].bar(
        x - width/2, fake_vals, width,
        label='Fake News (0)',
        color='red', alpha=0.75,
        edgecolor='black'
    )
    b2 = axes[1].bar(
        x + width/2, real_vals, width,
        label='Real News (1)',
        color='green', alpha=0.75,
        edgecolor='black'
    )
    axes[1].set_title(
        'Per Class Metrics',
        fontsize=13,
        fontweight='bold'
    )
    axes[1].set_ylabel('Score')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].set_ylim(0, 1.1)

    for bar, val in zip(b1, fake_vals):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center', va='bottom',
            fontsize=9
        )
    for bar, val in zip(b2, real_vals):
        axes[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center', va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig('static/metrics_visualization.png')
    plt.show()
    print("Saved to static/metrics_visualization.png")


# ============================================
# STEP 8: GENERATE REPORT
# ============================================

def generate_report(metrics, model_info,
                    tn, fp, fn, tp):
    print("\nGenerating Evaluation Report...")

    auc_str = f"{metrics['AUC-ROC']:.4f}" \
        if metrics['AUC-ROC'] else "N/A"

    report = f"""
{'='*60}
FAKE NEWS DETECTION SYSTEM
EVALUATION REPORT
{'='*60}

Model Information:
------------------
Model Name     : {model_info['model_name']}
Model Accuracy : {model_info['accuracy']:.4f}
                 ({model_info['accuracy']*100:.2f}%)

Dataset Information:
--------------------
Total Samples  : 16,989
Real News      : 11,326 (66.67%)
Fake News      :  5,663 (33.33%)
Train Split    : 80% (13,591 samples)
Test Split     : 20%  (3,398 samples)

Overall Performance Metrics:
-----------------------------
Accuracy       : {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)
Precision      : {metrics['Precision']:.4f}
Recall         : {metrics['Recall']:.4f}
F1-Score       : {metrics['F1-Score']:.4f}
AUC-ROC        : {auc_str}
MCC            : {metrics['MCC']:.4f}
Kappa Score    : {metrics['Kappa']:.4f}

Confusion Matrix:
-----------------
True Negative  (TN) : {tn}  (Correct Fake)
False Positive (FP) : {fp}  (Fake predicted as Real)
False Negative (FN) : {fn}  (Real predicted as Fake)
True Positive  (TP) : {tp}  (Correct Real)

Sensitivity (TPR) : {tp/(tp+fn):.4f}
Specificity (TNR) : {tn/(tn+fp):.4f}
False Pos Rate    : {fp/(fp+tn):.4f}
False Neg Rate    : {fn/(fn+tp):.4f}

Per Class Metrics:
------------------
Class         Precision    Recall    F1-Score
Fake News     {metrics['Precision (Fake)']:.4f}        {metrics['Recall (Fake)']:.4f}     {metrics['F1 (Fake)']:.4f}
Real News     {metrics['Precision (Real)']:.4f}        {metrics['Recall (Real)']:.4f}     {metrics['F1 (Real)']:.4f}

Feature Summary:
----------------
TF-IDF Features      : 15,462
URL Features         : 8
Domain Features      : 5
Title Features       : 9
Tweet Features       : 4
Total Features       : 15,488

Top Predictive Features:
------------------------
1. url_has_https      : 0.8011 (Real signal)
2. url_slashes        : 0.4706 (Real signal)
3. url_length         : 0.1860 (Real signal)
4. domain_trusted     : 0.1622 (Real signal)
5. title_question     : 0.1539 (Fake signal)
6. is_viral           : 0.1410 (Fake signal)
7. title_clickbait    : 0.0837 (Fake signal)

Generated Plots:
----------------
- static/detailed_confusion_matrix.png
- static/roc_curve_analysis.png
- static/precision_recall_curve.png
- static/metrics_visualization.png
- static/model_comparison.png
- static/feature_importance.png

{'='*60}
"""
    print(report)

    with open('reports/evaluation_report.txt', 'w') as f:
        f.write(report)
    print("Report saved to reports/evaluation_report.txt")

    return report


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("="*60)
    print("FAKE NEWS DETECTION - MODEL EVALUATION")
    print("="*60)

    # Step 1: Load everything
    (model, model_info,
     tfidf, feature_names,
     scaler, df) = load_everything()

    # Step 2: Prepare test data
    (X_train, X_test,
     y_train, y_test) = prepare_test_data(
        df, tfidf, feature_names, scaler
    )

    # Get predictions
    y_pred = model.predict(X_test)

    # Get scores
    try:
        y_scores = model.predict_proba(X_test)[:, 1]
    except:
        y_scores = model.decision_function(X_test)

    # Step 3: Comprehensive metrics
    metrics = comprehensive_metrics(
        y_test, y_pred, y_scores
    )

    # Step 4: Confusion matrix
    cm, tn, fp, fn, tp = detailed_confusion_matrix(
        y_test, y_pred
    )

    # Step 5: ROC curve
    auc, opt_threshold = roc_curve_analysis(
        y_test, y_scores
    )

    # Step 6: Precision recall
    ap, best_thr = precision_recall_analysis(
        y_test, y_scores
    )

    # Step 7: Visualize metrics
    visualize_metrics(metrics)

    # Step 8: Generate report
    generate_report(
        metrics, model_info,
        tn, fp, fn, tp
    )

    print("\n")
    print("="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"Best Model    : {model_info['model_name']}")
    print(f"Accuracy      : {model_info['accuracy']*100:.2f}%")
    print(f"AUC-ROC       : {metrics['AUC-ROC']:.4f}")
    print(f"F1 (Fake)     : {metrics['F1 (Fake)']:.4f}")
    print(f"F1 (Real)     : {metrics['F1 (Real)']:.4f}")
    print(f"\nAll plots saved to static/")
    print(f"Report saved to reports/evaluation_report.txt")