import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import category_encoders as ce  # Note: pip install category-encoders

# --- 1. CARGA Y LIMPIEZA INICIAL ---
# Asumiendo que el archivo se llama 'movies.csv'
df = pd.read_csv('data/movies.csv')

# Nombres a minúsculas
df.columns = df.columns.str.lower()

# Limpieza de Votos y Gross
df['votes'] = df['votes'].str.replace(',', '', regex=True).astype(float)
df['gross'] = df['gross'].str.replace(r'\$|M', '', regex=True).astype(float)

# Limpieza de Géneros (se guarda el primero para Encoding, y la lista para análisis)
df['genre_list'] = df['genre'].str.replace(r'\\n|\n', '', regex=True).str.strip().str.split(', ')
df['main_genre'] = df['genre_list'].str[0] # Usamos el primer género para encoding simple

# Limpieza de One-line
df["one-line"] = df["one-line"].str.replace(r'\\n|\n', ' ', regex=True).str.strip()

# --- 2. MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN ---
cols_num = ['rating', 'votes', 'runtime', 'gross']
stats = df[cols_num].describe()
stats.to_csv('medidas_estadisticas.csv')
print("Estadísticas guardadas en medidas_estadisticas.csv")

# --- 3. MEDIDAS DE POSICIÓN Y OUTLIERS ---
# Detectar y eliminar outliers en 'votes' (ejemplo) usando IQR
Q1 = df['votes'].quantile(0.25)
Q3 = df['votes'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df['votes'] < (Q1 - 1.5 * IQR)) | (df['votes'] > (Q3 + 1.5 * IQR)))]

# Visualización de Outliers (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[cols_num])
plt.title("Detección de Outliers")
plt.savefig('boxplot_outliers.png')
plt.close()

# --- 4. HISTOGRAMAS (DISTRIBUCIÓN) ---
df[cols_num].hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribución de Columnas")
plt.savefig('histogramas_distribucion.png')
plt.close()

# --- 5. GRÁFICO DE DISPERSIÓN ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='rating', y='votes', hue='rating')
plt.title("Relación Rating vs Votes")
plt.savefig('dispersion_rating_votes.png')
plt.close()

# --- 6. TRANSFORMACIONES DE COLUMNAS ---

# A. One Hot Encoding (Para géneros principales)
# Seleccionamos los 5 géneros más comunes para no crear demasiadas columnas
top_genres = df['main_genre'].value_counts().nlargest(5).index
df_ohe = pd.get_dummies(df['main_genre'].apply(lambda x: x if x in top_genres else 'Other'), prefix='genre')

# B. Label Encoding (Para una columna categórica, ej. 'year' si fuera ordinal)
le = LabelEncoder()
df['year_encoded'] = le.fit_transform(df['year'].astype(str))

# C. Binary Encoding (Útil para columnas con muchas categorías como 'main_genre')
encoder = ce.BinaryEncoder(cols=['main_genre'])
df_binary = encoder.fit_transform(df['main_genre'])

# --- 7. CORRELACIÓN Y ELIMINACIÓN ---
plt.figure(figsize=(8, 6))
corr_matrix = df[cols_num].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.savefig('correlacion.png')
plt.close()
# Si runtime y rating estuvieran correlacionados > 0.9, eliminaríamos una.

# --- 8. ESCALADO Y TRANSFORMACIÓN LOGARÍTMICA ---

# Transformación Logarítmica (Para corregir sesgo en 'votes' o 'gross')
# Usamos log1p (log(1+x)) para manejar ceros
df['votes_log'] = np.log1p(df['votes'])

# Standard Scaling (Media 0, Desviación 1)
scaler = StandardScaler()
df[['rating_scaled', 'runtime_scaled']] = scaler.fit_transform(df[['rating', 'runtime']].fillna(0))

print("Proceso completado. Imágenes y archivos guardados en la carpeta actual.")