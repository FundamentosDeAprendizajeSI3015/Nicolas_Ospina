import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import os

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score,
    adjusted_mutual_info_score
)
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prep(archivo, drop_cols):
    """Load CSV, drop specified columns, return X_scaled, y, features, df_clean."""
    df = pd.read_csv(archivo).dropna()
    features = df.drop(drop_cols, axis=1).columns.to_list()
    df_clean = df[features + ['label']].dropna()
    X = df_clean[features].values
    y = df_clean['label'].values
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y, features, df_clean


def umap_embed(X_scaled, random_state=42):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        n_components=2, random_state=random_state)
    return reducer.fit_transform(X_scaled)


def pca_embed(X_scaled):
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X_scaled)


# ── KMeans helpers ─────────────────────────────────────────────────────────────

def kmeans_elbow_silhouette(X_scaled, k_range=range(2, 11)):
    """Return inertias and silhouette scores for a range of k."""
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return list(k_range), inertias, silhouettes


def fit_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


# ── DBSCAN helpers ─────────────────────────────────────────────────────────────

def dbscan_param_search(X_scaled, eps_values, min_samples_values):
    """
    Grid-search over eps x min_samples.
    Returns a DataFrame with n_clusters, noise_ratio, silhouette (when valid).
    """
    records = []
    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)
            sil = np.nan
            if n_clusters >= 2 and noise_ratio < 0.9:
                mask = labels != -1
                if mask.sum() > n_clusters:
                    sil = silhouette_score(X_scaled[mask], labels[mask])
            records.append({
                'eps': eps, 'min_samples': ms,
                'n_clusters': n_clusters,
                'noise_ratio': round(noise_ratio, 3),
                'silhouette': round(sil, 4) if not np.isnan(sil) else np.nan
            })
    return pd.DataFrame(records)


def best_dbscan_params(param_df):
    """Pick the row with highest silhouette among runs with 2–10 clusters."""
    valid = param_df[(param_df['n_clusters'] >= 2) &
                     (param_df['n_clusters'] <= 10) &
                     (param_df['noise_ratio'] < 0.4)]
    if valid.empty:
        # fallback: least noise
        valid = param_df[param_df['n_clusters'] >= 2]
    if valid.empty:
        return param_df.iloc[0]
    return valid.loc[valid['silhouette'].idxmax()]


def fit_dbscan(X_scaled, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return labels


# ── Evaluation ─────────────────────────────────────────────────────────────────

def cluster_metrics(X_scaled, pred_labels, true_labels, name=""):
    mask = pred_labels != -1          # exclude noise for internal metrics
    n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    noise_pct   = 100 * np.sum(pred_labels == -1) / len(pred_labels)

    sil  = silhouette_score(X_scaled[mask], pred_labels[mask]) if n_clusters >= 2 and mask.sum() > n_clusters else np.nan
    db_i = davies_bouldin_score(X_scaled[mask], pred_labels[mask]) if n_clusters >= 2 and mask.sum() > n_clusters else np.nan
    ari  = adjusted_rand_score(true_labels[mask], pred_labels[mask])
    ami  = adjusted_mutual_info_score(true_labels[mask], pred_labels[mask])

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Clusters found  : {n_clusters}")
    print(f"  Noise points    : {noise_pct:.1f}%")
    print(f"  Silhouette      : {sil:.4f}" if not np.isnan(sil) else "  Silhouette      : N/A")
    print(f"  Davies-Bouldin  : {db_i:.4f}" if not np.isnan(db_i) else "  Davies-Bouldin  : N/A")
    print(f"  ARI (vs label)  : {ari:.4f}")
    print(f"  AMI (vs label)  : {ami:.4f}")
    return dict(name=name, n_clusters=n_clusters, noise_pct=noise_pct,
                silhouette=sil, davies_bouldin=db_i, ARI=ari, AMI=ami)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_elbow_silhouette(ks, inertias, silhouettes, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ks, inertias, 'o-', color='steelblue', linewidth=2)
    axes[0].set_title("Elbow Curve", fontweight='bold')
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")

    axes[1].plot(ks, silhouettes, 'o-', color='tomato', linewidth=2)
    axes[1].axhline(max(silhouettes), color='gray', linestyle='--', linewidth=0.8)
    axes[1].set_title("Silhouette Score vs k", fontweight='bold')
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")

    plt.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


def plot_clusters_2d(embedding, labels, title, ax, noise_label=-1):
    unique = sorted(set(labels))
    palette = sns.color_palette("tab10", len([l for l in unique if l != noise_label]))
    color_idx = 0
    color_map = {}
    for lbl in unique:
        if lbl == noise_label:
            color_map[lbl] = (0.5, 0.5, 0.5)   # grey for noise
        else:
            color_map[lbl] = palette[color_idx]
            color_idx += 1

    for lbl in unique:
        mask = labels == lbl
        lbl_name = "Noise" if lbl == noise_label else f"Cluster {lbl}"
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[color_map[lbl]], s=15, alpha=0.6,
                   label=lbl_name, linewidths=0)

    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=8, markerscale=1.5, framealpha=0.85)


def plot_true_labels_2d(embedding, y, ax):
    unique = sorted(set(y))
    palette = sns.color_palette("Set2", len(unique))
    for i, lbl in enumerate(unique):
        mask = y == lbl
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[palette[i]], s=15, alpha=0.6,
                   label=str(lbl), linewidths=0)
    ax.set_title("Ground Truth Labels", fontweight='bold', fontsize=11)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(title="Label", fontsize=8, markerscale=1.5, framealpha=0.85)


def plot_dbscan_heatmap(param_df, metric, title, ax):
    pivot = param_df.pivot(index='min_samples', columns='eps', values=metric)
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f", cmap='YlGnBu',
                cbar_kws={'label': metric}, linewidths=0.4)
    ax.set_title(title, fontweight='bold', fontsize=11)


def summary_bar(metrics_list, save_path):
    """Bar chart comparing ARI and Silhouette across methods."""
    names  = [m['name']      for m in metrics_list]
    aris   = [m['ARI']       for m in metrics_list]
    sils   = [m['silhouette'] if not np.isnan(m['silhouette']) else 0 for m in metrics_list]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, aris, width, label='ARI',       color='steelblue',  alpha=0.85)
    ax.bar(x + width/2, sils, width, label='Silhouette', color='darkorange', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_ylabel("Score"); ax.set_title("Clustering Performance Summary", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE — runs for one dataset
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering_pipeline(archivo, drop_cols, tag, best_k=None,
                             eps_values=None, min_samples_values=None):
    print(f"\n{'═'*60}")
    print(f"  DATASET: {archivo}  (tag={tag})")
    print(f"{'═'*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    X_scaled, y, features, df_clean = load_and_prep(archivo, drop_cols)
    embedding = umap_embed(X_scaled)

    # ── KMeans: elbow / silhouette ────────────────────────────────────────────
    print("\n[KMeans] Computing elbow & silhouette curves …")
    ks, inertias, silhouettes = kmeans_elbow_silhouette(X_scaled)
    plot_elbow_silhouette(ks, inertias, silhouettes,
                          f"KMeans — {tag}",
                          f"results/kmeans_elbow_{tag}.png")

    if best_k is None:
        best_k = ks[np.argmax(silhouettes)]
    print(f"  ➜ Using k = {best_k}")

    km_labels, km_model = fit_kmeans(X_scaled, best_k)
    km_metrics = cluster_metrics(X_scaled, km_labels, y, name=f"KMeans k={best_k} [{tag}]")

    # ── DBSCAN: param search ──────────────────────────────────────────────────
    if eps_values is None:
        eps_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if min_samples_values is None:
        min_samples_values = [3, 5, 10, 15]

    print("\n[DBSCAN] Parameter grid search …")
    param_df = dbscan_param_search(X_scaled, eps_values, min_samples_values)
    print(param_df.to_string(index=False))

    best_row = best_dbscan_params(param_df)
    best_eps = best_row['eps']
    best_ms  = int(best_row['min_samples'])
    print(f"\n  ➜ Best DBSCAN params: eps={best_eps}, min_samples={best_ms}")

    db_labels = fit_dbscan(X_scaled, best_eps, best_ms)
    db_metrics = cluster_metrics(X_scaled, db_labels, y, name=f"DBSCAN eps={best_eps} ms={best_ms} [{tag}]")

    # ── Visualisation panel ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    # Row 0: ground truth | kmeans | dbscan
    plot_true_labels_2d(embedding, y, axes[0, 0])
    plot_clusters_2d(embedding, km_labels,  f"KMeans  k={best_k}", axes[0, 1])
    plot_clusters_2d(embedding, db_labels,  f"DBSCAN  eps={best_eps} ms={best_ms}", axes[0, 2])

    # Row 1: elbow/sil inline, dbscan heatmaps
    axes[1, 0].plot(ks, inertias, 'o-', color='steelblue', linewidth=2)
    axes[1, 0].set_title("KMeans Elbow Curve", fontweight='bold')
    axes[1, 0].set_xlabel("k"); axes[1, 0].set_ylabel("Inertia")

    axes[1, 1].plot(ks, silhouettes, 'o-', color='tomato', linewidth=2)
    axes[1, 1].axvline(best_k, color='gray', linestyle='--', linewidth=1, label=f'k={best_k}')
    axes[1, 1].set_title("KMeans Silhouette vs k", fontweight='bold')
    axes[1, 1].set_xlabel("k"); axes[1, 1].set_ylabel("Silhouette")
    axes[1, 1].legend()

    plot_dbscan_heatmap(param_df, 'silhouette', "DBSCAN Silhouette Grid", axes[1, 2])

    plt.suptitle(f"Clustering Analysis — {tag}", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(f"results/clustering_panel_{tag}.png", bbox_inches='tight')
    plt.close()
    print(f"  Saved → results/clustering_panel_{tag}.png")

    # ── DBSCAN noise map ───────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    plot_dbscan_heatmap(param_df, 'n_clusters',   "DBSCAN n_clusters Grid",  axes2[0])
    plot_dbscan_heatmap(param_df, 'noise_ratio',  "DBSCAN Noise Ratio Grid", axes2[1])
    plt.suptitle(f"DBSCAN Parameter Sweep — {tag}", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig2.savefig(f"results/dbscan_param_sweep_{tag}.png", bbox_inches='tight')
    plt.close()
    print(f"  Saved → results/dbscan_param_sweep_{tag}.png")

    # ── Feature-level cluster profiles (KMeans) ────────────────────────────────
    df_clean = df_clean.copy()
    df_clean['km_cluster'] = km_labels
    cluster_profile = df_clean.groupby('km_cluster')[features].mean()

    fig3, ax3 = plt.subplots(figsize=(max(10, len(features)*0.8 + 2), 4))
    cluster_profile.T.plot(kind='bar', ax=ax3, colormap='tab10', width=0.75, edgecolor='white')
    ax3.set_title(f"KMeans Cluster Feature Profiles — {tag}", fontweight='bold')
    ax3.set_xlabel("Feature"); ax3.set_ylabel("Mean (scaled space ≈)")
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(title='Cluster', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    fig3.savefig(f"results/kmeans_profiles_{tag}.png", bbox_inches='tight')
    plt.close()
    print(f"  Saved → results/kmeans_profiles_{tag}.png")

    return km_metrics, db_metrics


# ══════════════════════════════════════════════════════════════════════════════
# RUN BOTH DATASETS
# ══════════════════════════════════════════════════════════════════════════════

all_metrics = []

# ── Dataset 1: realista ────────────────────────────────────────────────────────
km1, db1 = run_clustering_pipeline(
    archivo            = 'dataset_sintetico_FIRE_UdeA_realista.csv',
    drop_cols          = ['label', 'unidad'],
    tag                = 'realista',
    best_k             = None,           # auto from silhouette
    eps_values         = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    min_samples_values = [3, 5, 10, 15],
)
all_metrics += [km1, db1]

# ── Dataset 2: base ────────────────────────────────────────────────────────────
km2, db2 = run_clustering_pipeline(
    archivo            = 'dataset_sintetico_FIRE_UdeA.csv',
    drop_cols          = ['label'],
    tag                = 'base',
    best_k             = None,
    eps_values         = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
    min_samples_values = [3, 5, 10, 15],
)
all_metrics += [km2, db2]

# ── Global summary ─────────────────────────────────────────────────────────────
summary_bar(all_metrics, "results/clustering_summary.png")

summary_df = pd.DataFrame(all_metrics)
summary_df.to_csv("results/clustering_metrics_summary.csv", index=False)
print("\n[✓] All results saved to results/")
print(summary_df.to_string(index=False))