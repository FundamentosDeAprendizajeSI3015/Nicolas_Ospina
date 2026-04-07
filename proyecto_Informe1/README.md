# Proyecto Informe 1: Reconocimiento de Géneros Musicales (GTZAN)

Este directorio contiene el pipeline inicial para el análisis y procesamiento de características extraídas de señales de audio, utilizando el dataset **GTZAN (Music Genre Classification)**.

## Objetivos
Implementar un flujo de trabajo que permita explorar la estructura de los datos musicales, visualizar sus propiedades acústicas y reducir su dimensionalidad para identificar patrones por género.

## Metodología del Pipeline (`Pipeline.py`)

1. **Exploración Estadística**: Cálculo de métricas clave (media, desviación estándar, rangos) para las características de audio.
2. **Análisis de Balance**: Verificación de la distribución de las 10 clases de géneros musicales.
3. **Ingeniería de Características**:
   - Análisis de **Brillo (Spectral Centroid)** y **Tempo** por género.
   - **Filtro de Redundancia**: Identificación y eliminación de características con correlación > 0.9 para evitar el sesgo de multicolinealidad.
4. **Procesamiento de Audio**:
   - Generación de **Mel-Spectrograms** utilizando la librería `librosa` para comparar visualmente géneros como *Metal* vs. *Classical*.
5. **Reducción de Dimensionalidad**:
   - Implementación de **t-SNE** (t-Distributed Stochastic Neighbor Embedding) para proyectar el espacio de características en 2D, permitiendo visualizar la "separabilidad" de los géneros.

## Archivos y Estructura
- `Pipeline.py`: Script principal con el flujo de análisis.
- `Data/`:
    - `features_30_sec.csv`: Dataset con características promediadas en ventanas de 30 segundos.
    - `genres_original/`: Contiene los archivos de audio original (.wav) organizados por género.
- `pipeline_results/`: Carpeta que almacena las visualizaciones generadas:
    - `*spectrogram.jpg`: Espectrogramas Mel.
    - `t-SNE.jpg`: Mapeo bidimensional de los datos.
    - `Matriz_correlacion.jpg`: Visualización de la relación entre variables acústicas.

##  Requisitos
- Python 3.x
- `pandas`, `seaborn`, `matplotlib`, `numpy`, `librosa`, `sklearn`
