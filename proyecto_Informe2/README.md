# Proyecto Informe 2: Agrupamiento No Supervisado vs Clasificación Supervisada

En esta fase del proyecto, el enfoque se desplaza hacia el aprendizaje no supervisado (**Clustering**) y su comparación con modelos de clasificación supervisada para validar la coherencia de las etiquetas de género musical.

## Objetivos
1. Determinar si los géneros musicales definidos originalmente se agrupan de forma natural mediante algoritmos de clustering.
2. Evaluar la capacidad predictiva de un modelo cuando se entrena con etiquetas generadas por clustering vs. etiquetas reales.

##  Análisis Técnico (`clustering_supervised_learning.py`)

### 1. Algoritmos de Clustering Implementados
- **K-Means**: Agrupamiento basado en centroides (10 clusters).
- **DBSCAN**: Identificación de grupos por densidad de puntos, útil para detectar "ruido" o canciones atípicas.
- **Fuzzy C-Means**: Agrupamiento difuso donde cada muestra tiene un grado de pertenencia a múltiples grupos.

### 2. Evaluación del Clustering
- **Análisis de Proporciones**: Gráficos de barras apiladas que muestran la composición de géneros reales dentro de cada cluster detectado.
- **Métricas Externas**: Uso del **Adjusted Rand Index (ARI)** para medir la similitud entre los clusters y las etiquetas de género originales.
- **PCA (Principal Component Analysis)**: Reducción a 2 componentes para la visualización espacial de los clusters encontrados.

### 3. Experimentos con Random Forest
Se entrenan cuatro modelos de **Random Forest** (200 estimadores) variando el conjunto de etiquetas de entrenamiento:
- **Baseline**: Entrenamiento con etiquetas reales (Original Labels).
- **Cluster Labeled**: Entrenamiento con etiquetas de K-Means, Fuzzy C-Means y DBSCAN.

Se reportan métricas de **Accuracy**, **Classification Report** y **Matrices de Confusión** para cada caso, además de un análisis de **Importancia de Características (Feature Importance)**.

## Resultados Clave
Los resultados se almacenan en la carpeta `/results`:
- `rf_accuracy_summary.jpg`: Comparativa de precisión entre los diferentes sets de etiquetas.
- `*_label_comparison.jpg`: Distribución de géneros según el agrupamiento.
- `rf_confusion_*.jpg`: Matrices de confusión detalladas.

## Requisitos adicionales
Además de los requisitos del Informe 1, se requiere la librería `scikit-fuzzy`.
