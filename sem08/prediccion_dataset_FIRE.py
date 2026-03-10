import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, RocCurveDisplay)

archivo = 'dataset_sintetico_FIRE_UdeA.csv'
df = pd.read_csv(archivo) 

df = df.dropna()

# ── 1. Prep ───────────────────────────────────────────────────────────────────
# features = df.drop(["label","unidad"], axis=1).columns.to_list()
features = df.drop(["label"], axis=1).columns.to_list()

df_clean = df[features + ['label']].dropna()
X = df_clean[features].values
y = df_clean['label'].values

X_scaled = StandardScaler().fit_transform(X)

# ── 2. Correlation Matrix ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

corr = df_clean[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr, mask=mask, ax=axes[0],
    annot=True, fmt=".2f", cmap="coolwarm", center=0,
    square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 9}
)
axes[0].set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# ── 3. UMAP ───────────────────────────────────────────────────────────────────
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X_scaled)

unique_labels = sorted(np.unique(y))
palette = sns.color_palette("tab10", len(unique_labels))
color_map = dict(zip(unique_labels, palette))
colors = [color_map[lbl] for lbl in y]

axes[1].scatter(embedding[:, 0], embedding[:, 1],
                c=colors, s=18, alpha=0.7, linewidths=0)
axes[1].set_title("UMAP Projection", fontsize=14, fontweight="bold", pad=12)
axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")

legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[lbl], markersize=8, label=str(lbl))
    for lbl in unique_labels
]
axes[1].legend(handles=legend_handles, title="Label",
               framealpha=0.9, fontsize=9)

plt.suptitle("Exploratory Analysis", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("results/umap_correlation_matrix.png")
plt.show()



# ── Boxplots ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for i, feat in enumerate(features):
    sns.boxplot(
        data=df_clean, x='label', y=feat,
        palette='coolwarm', width=0.5,
        flierprops=dict(marker='o', markersize=3, alpha=0.4),
        ax=axes[i]
    )
    sns.stripplot(
        data=df_clean, x='label', y=feat,
        color='black', size=2.5, alpha=0.25, jitter=True, ax=axes[i]
    )
    axes[i].set_title(feat, fontweight='bold', fontsize=11)
    axes[i].set_xlabel('Label'); axes[i].set_ylabel('')

axes[-1].set_visible(False)  # hide empty 8th subplot

plt.suptitle("Feature Distributions by Label", fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("results/boxplots.png")
plt.show()

# ── 1. Prep ───────────────────────────────────────────────────────────────────
features = df.drop(["label"], axis=1).columns.to_list()

df_clean = df[features + ['label']].dropna()
X = df_clean[features]
y = df_clean['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── 2. Pipeline ───────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gb', GradientBoostingClassifier(random_state=42))
])

# ── 3. Grid Search ────────────────────────────────────────────────────────────
param_grid = {
    'gb__n_estimators':  [100, 200],
    'gb__learning_rate': [0.05, 0.1],
    'gb__max_depth':     [3, 4],
    'gb__subsample':     [0.8, 1.0],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=cv, scoring='roc_auc',
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best params: ", grid_search.best_params_)
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
y_pred      = best_model.predict(X_test)
y_prob      = best_model.predict_proba(X_test)[:, 1]

print(f"\nTest Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Test ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── 5. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            linewidths=0.5, cbar=False)
axes[0].set_title("Confusion Matrix", fontweight="bold")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

# ROC curve
RocCurveDisplay.from_estimator(best_model, X_test, y_test, ax=axes[1])
axes[1].plot([0,1],[0,1],'k--', linewidth=0.8)
axes[1].set_title("ROC Curve", fontweight="bold")

# Feature importance
gb = best_model.named_steps['gb']
importances = pd.Series(gb.feature_importances_, index=features).sort_values()

importances.plot(kind='barh', ax=axes[2], color='steelblue', edgecolor='white')
axes[2].set_title("Feature Importances", fontweight="bold")
axes[2].set_xlabel("Importance")
axes[2].axvline(importances.mean(), color='red', linestyle='--',
                linewidth=0.9, label='mean')
axes[2].legend(fontsize=9)

plt.suptitle("Gradient Boosting — Results", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("results/model_performance.png")
plt.show()