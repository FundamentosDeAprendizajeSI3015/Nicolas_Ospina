import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 1. Load and Prepare Data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

# Set visual style
sns.set_theme(style="whitegrid")

# 2. Pairplot (Feature Relationships)
# This shows how features relate to each other per species
sns.pairplot(df, hue='species', palette='viridis', height=2.5)
plt.subplots_adjust(top=0.95)
plt.gcf().suptitle('Iris Feature Pairplot')
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(8, 6))
# Exclude the string column for correlation
corr = df.drop(columns=['species']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# 4. PCA (Dimensionality Reduction)
pca = PCA(n_components=2)
components = pca.fit_transform(iris.data)
pca_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
pca_df['species'] = df['species']



plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, s=100, palette='magma')
plt.title(f'PCA: Explained Variance = {sum(pca.explained_variance_ratio_):.2%}')
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.show()

# 5. Boxplots (Feature Distribution)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, col in enumerate(iris.feature_names):
    sns.boxplot(ax=axes[i//2, i%2], x='species', y=col, data=df, palette='Set2')
    axes[i//2, i%2].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()