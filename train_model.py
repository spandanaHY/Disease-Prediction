import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import os

os.makedirs('static/graphs', exist_ok=True)

def train_disease(name, df, target_col, color):
    print(f"\n{'='*50}")
    print(f"  Training {name} Models")
    print(f"{'='*50}")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    print(f"  Dataset: {len(df)} rows | {X.shape[1]} features")

    # Preprocessing
    print("\n  [1/4] Preprocessing data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    print(f"        Train: {len(X_train)} | Test: {len(X_test)}")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree':       DecisionTreeClassifier(),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes':         GaussianNB(),
        'SVM':                 SVC(probability=True)
    }

    print("\n  [2/4] Training models...\n")
    acc = {}

    # Progress bar over models
    with tqdm(total=len(models), desc="  Models", ncols=60,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        for mname, model in models.items():
            model.fit(X_train, y_train)
            acc[mname] = accuracy_score(y_test, model.predict(X_test))
            pbar.set_postfix_str(f"{mname}: {acc[mname]*100:.1f}%")
            pbar.update(1)
            time.sleep(0.3)

    print("\n   Results:")
    for mname, a in acc.items():
        bar = "█" * int(a * 30)
        print(f"     {mname:<22} {bar:<30} {a*100:.2f}%")

    best_name = max(acc, key=acc.get)
    best_model = models[best_name]
    print(f"\n   Best Model: {best_name} ({acc[best_name]*100:.2f}%)")

    # Save model
    print("\n  [3/4] Saving model...")
    prefix = name.lower().replace(" ", "_")
    pickle.dump(best_model, open(f'{prefix}_model.pkl', 'wb'))
    pickle.dump(scaler,     open(f'{prefix}_scaler.pkl', 'wb'))
    time.sleep(0.3)
    print("         Model saved!")

    # Graphs
    print("\n  [4/4] Generating graphs...")
    with tqdm(total=3, desc="  Graphs", ncols=60,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:

        # Accuracy graph
        plt.figure(figsize=(10,5))
        plt.bar(acc.keys(), [v*100 for v in acc.values()], color=color)
        plt.title(f'{name} - Model Accuracy Comparison')
        plt.ylabel('Accuracy (%)')
        plt.ylim(50, 100)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f'static/graphs/{prefix}_accuracy.png')
        plt.close()
        pbar.set_postfix_str("Accuracy graph")
        pbar.update(1)
        time.sleep(0.2)

        # Confusion matrix
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        labels = ['No '+name, name]
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if color=='steelblue' else 'Reds',
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'{name} Confusion Matrix ({best_name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'static/graphs/{prefix}_confusion.png')
        plt.close()
        pbar.set_postfix_str("Confusion matrix")
        pbar.update(1)
        time.sleep(0.2)

        # Feature importance
        plt.figure(figsize=(8,5))
        if hasattr(best_model, 'feature_importances_'):
            imp = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            imp = np.abs(best_model.coef_[0])
        else:
            imp = np.ones(X.shape[1])
        plt.barh(X.columns, imp, color=color)
        plt.title(f'{name} - Feature Importance ({best_name})')
        plt.xlabel('Score')
        plt.tight_layout()
        plt.savefig(f'static/graphs/{prefix}_features.png')
        plt.close()
        pbar.set_postfix_str("Feature importance")
        pbar.update(1)
        time.sleep(0.2)

    print("         All graphs saved!")
    return best_name, acc[best_name]

# ── MAIN ──────────────────────────────────────────────────
print("\n" + "🔬 DISEASE PREDICTION SYSTEM — MODEL TRAINING".center(60))
print("="*60)

df_d = pd.read_csv('datasets/diabetes.csv')
df_h = pd.read_csv('datasets/heart.csv')

d_best, d_acc = train_disease('Diabetes',     df_d, 'Outcome', 'steelblue')
h_best, h_acc = train_disease('Heart_Disease', df_h, 'target',  'tomato')

print("\n" + "="*60)
print(" TRAINING COMPLETE!")
print(f"   Diabetes     → Best: {d_best} ({d_acc*100:.2f}%)")
print(f"   Heart Disease → Best: {h_best} ({h_acc*100:.2f}%)")
print("="*60)
