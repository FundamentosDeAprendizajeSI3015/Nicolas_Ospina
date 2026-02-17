import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# librería para procesamiento de características en el audio
import librosa
import librosa.display

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# Cargar el dataset
df = pd.read_csv("data/features_30_sec.csv")

# --- Estadísitca descriptica por cada columna
mean_cols = [col for col in df.columns if 'mean' in col]
stats_summary = df[mean_cols].describe().T # Transpuesta para lectura

print("Key Feature Statistics:")
print(stats_summary[['mean', 'std', 'min', 'max']])


# --- Distribución de clases, para verificar el balance en los datos
plt.figure(figsize=(10, 5))
counts = df['label'].value_counts()
plt.bar(counts.index, counts.values, color='skyblue', edgecolor='black')
plt.title('Distribución por Géneros')
plt.ylabel('Número de muestras')
plt.xticks(rotation=45)
plt.savefig("pipeline_results/Distribución_de_Géneros.jpg")


# --- Observar el 'brillo' de la canción, definido por 'spectral_centroid_mean'
genre_means = df.groupby('label').mean(numeric_only=True)

plt.figure(figsize=(12, 6))
genre_means['spectral_centroid_mean'].sort_values().plot(kind='barh', color='teal')
plt.title('"Brillo" promedio (Spectral Centroid) por Género')
plt.xlabel('Frecuencia (Hz)')
plt.savefig("pipeline_results/Brillo_promedio.jpg")


# --- Visualizar como el tempo cambia por género
plt.figure(figsize=(16, 6))
sns.boxplot(x='label', y='tempo', data=df, palette='viridis')
plt.title('Distribución del tempo por Género', fontsize=15)
plt.xticks(rotation=45)

plt.savefig("pipeline_results/Tempo_por_genero.jpg")


# ---Matriz de correlación, para verificar linealidad entre las características
plt.figure(figsize=(12, 10))
corr = df.drop(columns=['filename', 'label']).corr()
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title('Matriz de Correlación')

plt.savefig("pipeline_results/Matriz_correlacion.jpg")


# --- Hallar las parejas de datos con una correlación mayor a 0.9 (Alta)
#Cacular la matriz de correlación
corr_matrix = df.corr(numeric_only=True)
# Filtrar por alta correlación
high_corr = corr_matrix.stack().reset_index()
high_corr.columns = ['Feature_1', 'Feature_2', 'Correlation']
high_corr = high_corr[(high_corr['Correlation'] > 0.9) & (high_corr['Feature_1'] != high_corr['Feature_2'])]

print("Highly Correlated (Redundant) Features:")
print(high_corr.sort_values(by='Correlation', ascending=False).head(10))

# Eliminar las columans con una alta correlación
df = df.drop(high_corr["Feature_2"].values, axis=1)

# ---Visualización de los espectogramas
def plot_spectrogram(genre, file_idx=0):
    path = f'Data/genres_original/{genre}/{genre}.0000{file_idx}.wav'
    y, sr = librosa.load(path)
    
    # Calcular el Spectograma a partir de un ejemplo de audio
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram: {genre.capitalize()}')
    plt.tight_layout()
    plt.savefig("pipeline_results/"+ genre +"_spectrogram.jpg")

# Comparación entre géneros con energías diferentes
plot_spectrogram('metal')
plot_spectrogram('classical')

# Reducción de dimensionalidad, para apreciar ditribución de los datos
random_state = 42

#ELiminamos las columnas no útiles para la reducción de dimensionalidad
X = df.drop(columns=['filename', 'label', 'length'])
y = df["label"]

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applicar la reducción de dimensionalida sobre los datos estandarizados
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, init='random')

X_tsne = tsne.fit_transform(X_scaled)


# 1. Crear un Dataframe temporal
# X_tsne es un arreglo con 2 columnas; se convierte a un DF y se añade los labels
tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['label'] = y.values 

# 2. Plottear el t-SNE
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='TSNE1', 
    y='TSNE2',
    hue='label',           # Agrupar los datos, para colorear cada etiqueta
    palette='hls',         
    data=tsne_df,
    legend='full',
    alpha=0.8
)

# 3. Embellecer la gráfica
plt.title('t-SNE: Genre', fontsize=15)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()
plt.savefig("pipeline_results/t-SNE.jpg")

le = LabelEncoder()
# Transformar los labels
# "jazz", "metal", "rock" -> 0, 1, 2...
y_encoded = le.fit_transform(df['label'])

# Verificar corrrespondencia numérica:
mapping = dict(zip(le.classes_, range(len(le.classes_))))
print(f"Genre Mapping: {mapping}")

# Aplicar label Encoder a los datos.
df["label"] = y_encoded