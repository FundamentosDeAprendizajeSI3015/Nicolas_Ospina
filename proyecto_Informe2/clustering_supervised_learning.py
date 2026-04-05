import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, adjusted_rand_score
)
import skfuzzy as fuzz

# ── reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = "Data/features_30_sec.csv"
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. LOAD & PRE-PROCESS  (same pipeline as Informe 1)
# =============================================================================
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")

# --- drop non-numeric / identifier columns ---
X_raw = df.drop(columns=["filename", "label", "length"])
y_true_labels = df["label"]

# --- encode true labels ---
le = LabelEncoder()
y_true = le.fit_transform(y_true_labels)
class_names = list(le.classes_)
N_CLASSES = len(class_names)          # 10 genres
print(f"Classes ({N_CLASSES}): {class_names}")

# --- remove highly-correlated features (corr > 0.9), as in Pipeline.py ---
corr_matrix = X_raw.corr(numeric_only=True)
high_corr    = corr_matrix.stack().reset_index()
high_corr.columns = ["Feature_1", "Feature_2", "Correlation"]
high_corr = high_corr[
    (high_corr["Correlation"] > 0.9) &
    (high_corr["Feature_1"] != high_corr["Feature_2"])
]
X_raw = X_raw.drop(columns=high_corr["Feature_2"].values, errors="ignore")
print(f"Features after corr-filter: {X_raw.shape[1]}")

# --- standardise ---
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# --- PCA to 2-D for visualisation ---
pca_2d  = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d    = pca_2d.fit_transform(X_scaled)

# --- PCA to 50-D for clustering (keeps more info) ---
pca_50  = PCA(n_components=50, random_state=RANDOM_STATE)
X_50    = pca_50.fit_transform(X_scaled)

PALETTE = sns.color_palette("hls", N_CLASSES)

# helper: map arbitrary cluster ids → genre names for plot titles
def _label_str(lbl, encoder=le):
    if lbl == -1:
        return "noise"
    return encoder.classes_[lbl % N_CLASSES]


# =============================================================================
# 2. CLUSTERING
# =============================================================================

# ── 2a.  K-Means ──────────────────────────────────────────────────────────────
print("\n--- K-Means ---")
kmeans      = KMeans(n_clusters=N_CLASSES, random_state=RANDOM_STATE, n_init=20)
y_kmeans    = kmeans.fit_predict(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                     c=y_kmeans, cmap="tab10", alpha=0.7, s=20)
legend_handles = [mpatches.Patch(color=f"C{i}", label=f"Cluster {i}")
                  for i in range(N_CLASSES)]
ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_title("K-Means Clustering (PCA 2D projection)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "kmeans_clusters.jpg"), dpi=150)
plt.close()
print(f"  ARI (vs true labels): {adjusted_rand_score(y_true, y_kmeans):.4f}")

# ── 2b.  DBSCAN  ──────────────────────────────────────────────────────────────
print("\n--- DBSCAN ---")
# eps tuned for standardised GTZAN; min_samples = 2*n_features rule-of-thumb
dbscan      = DBSCAN(eps=3.5, min_samples=5)
y_dbscan    = dbscan.fit_predict(X_50)   # use 50-PCA for speed & quality

n_dbscan_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise           = (y_dbscan == -1).sum()
print(f"  Clusters found: {n_dbscan_clusters}  |  Noise points: {n_noise}")

# colour map: noise = grey, clusters = hls palette
unique_ids  = sorted(set(y_dbscan))
color_map   = {}
palette_hls = sns.color_palette("hls", max(n_dbscan_clusters, 1))
c_idx       = 0
for uid in unique_ids:
    color_map[uid] = "grey" if uid == -1 else palette_hls[c_idx % len(palette_hls)]
    if uid != -1:
        c_idx += 1

colors_dbscan = [color_map[lbl] for lbl in y_dbscan]

fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors_dbscan, alpha=0.7, s=20)
handles = []
for uid in unique_ids:
    label = "noise" if uid == -1 else f"Cluster {uid}"
    handles.append(mpatches.Patch(color=color_map[uid], label=label))
ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
ax.set_title(f"DBSCAN Clustering  ({n_dbscan_clusters} clusters, PCA 2D projection)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "dbscan_clusters.jpg"), dpi=150)
plt.close()

# for DBSCAN: use only the number of clusters found (not per-point mapping)
print(f"  DBSCAN n_clusters used downstream: {n_dbscan_clusters}")

# ── 2c.  Fuzzy C-Means ────────────────────────────────────────────────────────
print("\n--- Fuzzy C-Means ---")
# skfuzzy expects shape (features, samples)
X_fcm = X_scaled.T.astype(np.float64)

cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
    X_fcm, c=N_CLASSES, m=2, error=0.005, maxiter=1000, init=None, seed=RANDOM_STATE
)
y_fuzzy = np.argmax(u, axis=0)   # hard assignment from max membership
print(f"  Fuzzy Partition Coefficient (FPC): {fpc:.4f}")
print(f"  ARI (vs true labels): {adjusted_rand_score(y_true, y_fuzzy):.4f}")

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                     c=y_fuzzy, cmap="tab10", alpha=0.7, s=20)
legend_handles = [mpatches.Patch(color=f"C{i}", label=f"Cluster {i}")
                  for i in range(N_CLASSES)]
ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")
ax.set_title("Fuzzy C-Means Clustering (PCA 2D projection)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fuzzy_cmeans_clusters.jpg"), dpi=150)
plt.close()


# =============================================================================
# 3. COMPARE CLUSTER TAGS vs  REAL TAGS  (bar plots)
# =============================================================================

def plot_label_comparison(y_cluster, cluster_name, results_dir):
    """
    Build a dataframe: for each sample, store (true_genre, cluster_id).
    Then for each cluster show the distribution of true genres inside it,
    and plot as a stacked bar chart.
    """
    df_cmp = pd.DataFrame({
        "true_genre"  : [class_names[t] for t in y_true],
        "cluster"     : y_cluster
    })

    # --- stacked bar: cluster  x  genre composition ---
    ct = pd.crosstab(df_cmp["cluster"], df_cmp["true_genre"])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)   # proportions

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # raw counts
    ct.plot(kind="bar", ax=axes[0], colormap="tab10", edgecolor="black", linewidth=0.4)
    axes[0].set_title(f"{cluster_name} — Genre count per cluster")
    axes[0].set_xlabel("Cluster ID"); axes[0].set_ylabel("# Samples")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Genre", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    # proportions
    ct_norm.plot(kind="bar", stacked=True, ax=axes[1],
                 colormap="tab10", edgecolor="black", linewidth=0.4)
    axes[1].set_title(f"{cluster_name} — Genre proportion per cluster")
    axes[1].set_xlabel("Cluster ID"); axes[1].set_ylabel("Proportion")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Genre", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    plt.suptitle(f"{cluster_name}: Cluster labels vs Real genre labels", fontsize=13)
    plt.tight_layout()
    fname = os.path.join(results_dir,
                         f"{cluster_name.lower().replace(' ', '_')}_label_comparison.jpg")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


print("\n--- Label comparison plots ---")
plot_label_comparison(y_kmeans,  "KMeans",  RESULTS_DIR)
plot_label_comparison(y_fuzzy,   "FuzzyCMeans", RESULTS_DIR)

# DBSCAN: use only n_clusters number (treat noise as its own bucket)
plot_label_comparison(y_dbscan, "DBSCAN", RESULTS_DIR)


# =============================================================================
# 4. RANDOM FOREST
#    4a – original genre labels
#    4b – KMeans cluster labels
#    4c – Fuzzy C-Means cluster labels
#    4d – DBSCAN cluster labels  (only n_dbscan_clusters classes used, noise=-1)
# =============================================================================

def run_random_forest(X, y, label_set_name, results_dir,
                      label_names=None, random_state=42):
    """
    Train / evaluate a Random Forest. Saves confusion matrix + prints report.
    Returns accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_features="sqrt",
        random_state=random_state, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  [{label_set_name}]  Test accuracy: {acc:.4f}")

    # Build label list that matches ONLY the classes present in the test split
    present_ids = sorted(set(y_test) | set(y_pred))
    if label_names is not None:
        # label_names[i] corresponds to class-id i (0-indexed)
        present_names = [label_names[i] for i in present_ids]
    else:
        present_names = [str(i) for i in present_ids]

    print(classification_report(
        y_test, y_pred,
        labels=present_ids,
        target_names=present_names
    ))

    # ── confusion matrix ──
    cm = confusion_matrix(y_test, y_pred, labels=present_ids)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set_title(f"Random Forest – Confusion Matrix\n({label_set_name})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fname = os.path.join(results_dir,
                         f"rf_confusion_{label_set_name.lower().replace(' ', '_')}.jpg")
    plt.savefig(fname, dpi=150)
    plt.close()

    # ── feature importance (top 20) ──
    importances = pd.Series(clf.feature_importances_, index=X_raw.columns)
    top20 = importances.nlargest(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    top20.sort_values().plot(kind="barh", color="steelblue", ax=ax)
    ax.set_title(f"Feature Importance (top 20) – {label_set_name}")
    ax.set_xlabel("Gini importance")
    plt.tight_layout()
    fname_fi = os.path.join(results_dir,
                            f"rf_feature_importance_{label_set_name.lower().replace(' ', '_')}.jpg")
    plt.savefig(fname_fi, dpi=150)
    plt.close()

    return acc


print("\n=== Random Forest Experiments ===")

# 4a – original labels
acc_orig = run_random_forest(
    X_scaled, y_true, "Original_Labels", RESULTS_DIR,
    label_names=class_names
)

# 4b – KMeans labels
km_names = [f"KM_{i}" for i in range(N_CLASSES)]
acc_km   = run_random_forest(
    X_scaled, y_kmeans, "KMeans_Labels", RESULTS_DIR,
    label_names=km_names
)

# 4c – Fuzzy C-Means labels
fcm_names = [f"FCM_{i}" for i in range(N_CLASSES)]
acc_fcm   = run_random_forest(
    X_scaled, y_fuzzy, "FuzzyCMeans_Labels", RESULTS_DIR,
    label_names=fcm_names
)

# 4d – DBSCAN labels (exclude noise=-1 for a cleaner RF)
mask_valid      = y_dbscan != -1
X_db_valid      = X_scaled[mask_valid]
y_db_valid      = y_dbscan[mask_valid]
db_unique_ids   = sorted(set(y_db_valid))
db_label_names  = [f"DB_{i}" for i in db_unique_ids]

# re-encode to 0..k-1 if DBSCAN ids are non-contiguous
db_le           = LabelEncoder()
y_db_encoded    = db_le.fit_transform(y_db_valid)

acc_db = run_random_forest(
    X_db_valid, y_db_encoded, "DBSCAN_Labels", RESULTS_DIR,
    label_names=db_label_names
)

# =============================================================================
# 5. ACCURACY SUMMARY BAR CHART
# =============================================================================
summary = {
    "Original\nLabels"  : acc_orig,
    "K-Means\nLabels"   : acc_km,
    "Fuzzy C-Means\nLabels": acc_fcm,
    f"DBSCAN\nLabels\n({n_dbscan_clusters} clusters)": acc_db,
}

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(summary.keys(), summary.values(),
              color=["steelblue", "coral", "mediumseagreen", "orchid"],
              edgecolor="black")
for bar, val in zip(bars, summary.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylim(0, 1.05)
ax.set_ylabel("Test Accuracy")
ax.set_title("Random Forest Accuracy – Original vs Clustering Labels")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "rf_accuracy_summary.jpg"), dpi=150)
plt.close()

print("\n=== All done! Results saved to:", RESULTS_DIR, "===")
print(f"  Original labels  RF accuracy : {acc_orig:.4f}")
print(f"  K-Means labels   RF accuracy : {acc_km:.4f}")
print(f"  FuzzyCMeans lbl  RF accuracy : {acc_fcm:.4f}")
print(f"  DBSCAN labels    RF accuracy : {acc_db:.4f}")
