import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import umap  # Ensure pip install umap-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & STYLE
# ─────────────────────────────────────────────────────────────────────────────
PALETTE     = ["#4C72B0", "#DD8452"]
CLR_CORRECT = "#2ecc71"
CLR_WRONG   = "#e74c3c"
BG_COLOR    = "#ffffff"
CMAP_DISC   = "tab10" # Good for handling varied cluster counts

# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING CLASSES (FCM & Subtractive)
# ─────────────────────────────────────────────────────────────────────────────

class SubtractiveClustering:
    def __init__(self, ra=0.5, rb=0.75, eps_upper=0.5, eps_lower=0.15):
        self.ra, self.rb = ra, rb
        self.eps_upper, self.eps_lower = eps_upper, eps_lower

    def fit(self, X):
        x_min, x_range = X.min(axis=0), X.max(axis=0) - X.min(axis=0)
        x_range[x_range == 0] = 1.0
        X_norm = (X - x_min) / x_range
        D = np.zeros(len(X_norm))
        for xi in X_norm:
            dist_sq = np.sum(((X_norm - xi) / (self.ra / 2)) ** 2, axis=1)
            D += np.exp(-dist_sq)
        centers_norm, D1_max = [], D.max()
        while True:
            idx = np.argmax(D)
            P_k, c_k = D[idx], X_norm[idx]
            ratio = P_k / D1_max
            if ratio > self.eps_upper:
                centers_norm.append(c_k)
                D -= P_k * np.exp(-np.sum(((X_norm - c_k) / (self.rb / 2)) ** 2, axis=1))
                D = np.clip(D, 0, None)
            else: break
        self.centers_ = np.array([c * x_range + x_min for c in centers_norm])
        self.n_clusters_ = len(self.centers_)
        return self

    def predict(self, X):
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in self.centers_])
        return np.argmin(dists, axis=0)

class FuzzyCMeans:
    def __init__(self, n_clusters=2, m=2.0, max_iter=150, tol=1e-5):
        self.n_clusters, self.m, self.max_iter, self.tol = n_clusters, m, max_iter, tol

    def fit(self, X):
        n = X.shape[0]
        U = np.random.dirichlet(np.ones(self.n_clusters), size=n)
        for _ in range(self.max_iter):
            U_old = U.copy()
            U_m = U ** self.m
            centers = (U_m.T @ X) / U_m.sum(axis=0)[:, None]
            dist = np.array([np.linalg.norm(X - c, axis=1) for c in centers]).T
            dist = np.fmax(dist, 1e-10)
            inv_dist = 1.0 / (dist ** (2 / (self.m - 1)))
            U = inv_dist / inv_dist.sum(axis=1, keepdims=True)
            if np.linalg.norm(U - U_old) < self.tol: break
        self.U_, self.centers_ = U, centers
        return self

    def predict(self):
        return np.argmax(self.U_, axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def align_labels(y_true, y_pred):
    acc_original = np.mean(y_pred == y_true)
    acc_swapped = np.mean((1 - y_pred) == y_true)
    if acc_swapped > acc_original:
        return 1 - y_pred, acc_swapped
    return y_pred, acc_original

def plot_clusters_comparison(X_umap, y_true, y_km, y_fcm, y_sub, tag, out_dir):
    """Generates a 4-panel comparison using UMAP coordinates."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    fig.patch.set_facecolor(BG_COLOR)
    
    titles = ["Ground Truth", "KMeans (Aligned)", "Fuzzy C-Means", "Subtractive"]
    labels_list = [y_true, y_km, y_fcm, y_sub]
    
    for ax, labels, title in zip(axes, labels_list, titles):
        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, 
                            cmap=CMAP_DISC, s=20, alpha=0.7)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.suptitle(f"UMAP Projection & Clustering Results ({tag})", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/umap_comparison_{tag}.png", bbox_inches="tight")
    plt.close()

def plot_evaluation_panel(y_true, y_pred, tag, algo_name, out_dir):
    # (Kept from previous version to maintain performance metrics)
    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor(BG_COLOR)
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)
    
    ax0 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=ax0, cmap="Blues", colorbar=False)
    ax0.set_title(f"CM: {algo_name}", fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 2])
    correct = np.sum(y_true == y_pred)
    ax2.pie([correct, len(y_true)-correct], labels=["Right", "Not Right"], 
            colors=[CLR_CORRECT, CLR_WRONG], autopct="%1.1f%%", startangle=90)
    ax2.set_title("Overall Precision", fontweight="bold")
    
    # ... (rest of evaluation panel logic)
    plt.savefig(f"{out_dir}/eval_{algo_name.lower().replace(' ', '_')}_{tag}.png", bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(file_path, drop_cols, tag):
    print(f"\n🚀 Running UMAP Pipeline: {tag}")
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(file_path).dropna()
    X_features = df.drop(columns=drop_cols, errors='ignore')
    X_scaled = StandardScaler().fit_transform(X_features)
    y_true = df['label'].values.astype(int)

    # 1. Dimensionality Reduction (UMAP)
    print("  > Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # 2. Algorithm Runs
    print("  > Clustering...")
    km_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X_scaled)
    km_aligned, _ = align_labels(y_true, km_labels)

    fcm_labels = FuzzyCMeans(n_clusters=2).fit(X_scaled).predict()
    fcm_aligned, _ = align_labels(y_true, fcm_labels)

    sub = SubtractiveClustering().fit(X_scaled)
    sub_labels = sub.predict(X_scaled)

    # 3. Generate Visuals
    plot_clusters_comparison(X_umap, y_true, km_aligned, fcm_aligned, sub_labels, tag, out_dir)
    plot_evaluation_panel(y_true, km_aligned, tag, "KMeans", out_dir)
    plot_evaluation_panel(y_true, fcm_aligned, tag, "Fuzzy C-Means", out_dir)

    print(f"✅ Finished {tag}. Check /{out_dir}")

# EXECUTION
datasets = [
    ("dataset_sintetico_FIRE_UdeA_realista.csv", ["label", "unidad"], "realista"),
    ("dataset_sintetico_FIRE_UdeA.csv", ["label"], "base")
]

for path, drop, tag in datasets:
    if os.path.exists(path):
        run_pipeline(path, drop, tag)