import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
)
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# STYLE & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
PALETTE      = ["#4C72B0", "#DD8452"]
CLR_CORRECT  = "#2ecc71"
CLR_WRONG    = "#e74c3c"
CMAP_CM      = LinearSegmentedColormap.from_list("cm_cmap", ["#f7fbff", "#2171b5"])
BG           = "#f8f9fa"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prep(archivo, drop_cols):
    df = pd.read_csv(archivo).dropna()
    features = [c for c in df.columns if c not in drop_cols]
    df_clean = df[features + ["label"]].dropna()
    X = df_clean[features].values
    y = df_clean["label"].values.astype(int)
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y, features

def align_labels(pred, true):
    """For k=2: swap cluster ids if that gives better accuracy."""
    acc_d = np.mean(pred == true)
    acc_f = np.mean((1 - pred) == true)
    if acc_f > acc_d:
        return 1 - pred, acc_f
    return pred, acc_d

def run_kmeans(X_scaled, y):
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    raw = km.fit_predict(X_scaled)
    aligned, acc = align_labels(raw, y)
    return aligned, acc

# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING (Simplified for KMeans)
# ══════════════════════════════════════════════════════════════════════════════

def plot_evaluation_panel(y_true, y_pred, tag, acc):
    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.3)
    
    # 1. Confusion Matrix
    ax0 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
    disp.plot(ax=ax0, cmap=CMAP_CM, colorbar=False)
    ax0.set_title("Confusion Matrix", fontweight="bold")

    # 2. Per-Label Accuracy
    ax1 = fig.add_subplot(gs[0, 1])
    for i in [0, 1]:
        m = y_true == i
        val = np.mean(y_pred[m] == y_true[m]) * 100
        ax1.bar(f"Label {i}", val, color=PALETTE[i])
        ax1.text(i, val + 1, f"{val:.1f}%", ha='center', fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.set_title("Accuracy per Class", fontweight="bold")

    # 3. Pie Chart
    ax2 = fig.add_subplot(gs[0, 2])
    correct = np.sum(y_true == y_pred)
    ax2.pie([correct, len(y_true)-correct], labels=["Correct", "Wrong"], 
           colors=[CLR_CORRECT, CLR_WRONG], autopct="%1.1f%%", startangle=90)
    ax2.set_title("Overall Correctness", fontweight="bold")

    # 4. Report Heatmap
    ax3 = fig.add_subplot(gs[0, 3])
    rep = classification_report(y_true, y_pred, output_dict=True)
    sns.heatmap(pd.DataFrame(rep).iloc[:-1, :2].T, annot=True, cmap="Blues", ax=ax3)
    ax3.set_title("Classification Metrics", fontweight="bold")

    plt.suptitle(f"KMeans Performance: {tag} (Total Acc: {acc*100:.2f}%)", 
                 fontsize=14, fontweight="bold", y=1.05)
    plt.savefig(f"results/eval_{tag}.png", bbox_inches="tight")
    plt.close()

def save_visuals(X_scaled, y, y_pred, tag):
    pca = PCA(n_components=2).fit_transform(X_scaled)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(pca[:, 0], pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    ax1.set_title("Ground Truth")
    
    ax2.scatter(pca[:, 0], pca[:, 1], c=y_pred, cmap='coolwarm', alpha=0.6)
    ax2.set_title("KMeans Clusters")
    
    plt.savefig(f"results/pca_{tag}.png")
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(archivo, drop_cols, tag):
    print(f"\nProcessing {tag}...")
    X_scaled, y, _ = load_and_prep(archivo, drop_cols)
    
    y_pred, acc = run_kmeans(X_scaled, y)
    
    # Metrics
    sil = silhouette_score(X_scaled, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    print(f"  Acc: {acc:.4f} | Sil: {sil:.4f} | ARI: {ari:.4f}")
    
    plot_evaluation_panel(y, y_pred, tag, acc)
    save_visuals(X_scaled, y, y_pred, tag)
    
    return {"tag": tag, "accuracy": acc, "silhouette": sil, "ARI": ari}

# Execute
results = []
# Ensure these files exist in your directory
datasets = [
    ("dataset_sintetico_FIRE_UdeA_realista.csv", ["label", "unidad"], "realista"),
    ("dataset_sintetico_FIRE_UdeA.csv", ["label"], "base")
]

for path, drop, tag in datasets:
    if os.path.exists(path):
        results.append(run_pipeline(path, drop, tag))

if results:
    summary = pd.DataFrame(results)
    summary.to_csv("results/kmeans_summary.csv", index=False)
    print("\n[✓] Done! Results saved in /results")
    print(summary)