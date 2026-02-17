# Fundamentos de Aprendizaje Automático - Proyectos

Este repositorio contiene los proyectos desarrollados durante el curso de Fundamentos de Aprendizaje Automático (SI3015) en el primer semestre de 2026.

**Autor:** Nicolás Ospina  
**Programa:** Ingeniería de Sistemas  
**Universidad:** Universidad EAFIT

---

## 📂 Estructura del Repositorio

```
Nicolas_Ospina/
├── sem02/          # Análisis exploratorio con Iris Dataset
├── sem03/          # Preprocesamiento de datos financieros
├── sem04/          # EDA avanzado con dataset de películas
└── gtzan/          # Pipeline de clasificación de géneros musicales
```

---

## 🎯 Proyectos Realizados

### 1️⃣ Semana 2: Análisis del Dataset Iris
**Archivo:** `sem02/ospina_nicolas_iris_analysis.py`

Análisis exploratorio del clásico dataset Iris utilizando técnicas de visualización y reducción de dimensionalidad.

**Técnicas aplicadas:**
- **Pairplot:** Visualización de relaciones entre características por especie
- **Matriz de correlación:** Análisis de dependencias entre variables
- **PCA (Análisis de Componentes Principales):** Reducción a 2 dimensiones con varianza explicada
- **Boxplots:** Distribución de características por especie

**Librerías:** pandas, seaborn, matplotlib, scikit-learn

---

### 2️⃣ Semana 3: Preprocesamiento de Datos Fintech
**Archivo:** `sem03/Fintech1000/lab_fintech_sintetico_2025_1000.py`

Pipeline de preprocesamiento para datos sintéticos de empresas fintech, preparando los datos para modelos de machine learning.

**Procesos implementados:**
- **Limpieza de datos:** Manejo de valores nulos en variables numéricas y categóricas
- **Feature engineering:** Cálculo de retornos y log-retornos para precios
- **Encoding:** One-hot encoding para variables categóricas
- **Split temporal:** División train/test basada en fecha (antes/después de sept-2025)
- **Normalización:** StandardScaler aplicado a variables numéricas
- **Exportación:** Guardado en formato Parquet para eficiencia

**Librerías:** pandas, numpy, scikit-learn

---

### 3️⃣ Semana 4: EDA Completo - Dataset de Películas
**Archivo:** `sem04/EDA-NicolasOspina.py`

Análisis exploratorio exhaustivo de un dataset de películas con múltiples técnicas de preprocesamiento y visualización.

**Análisis estadístico:**
- Medidas de tendencia central y dispersión
- Detección y eliminación de outliers usando IQR
- Histogramas de distribución
- Gráficos de dispersión (Rating vs Votes)

**Transformaciones aplicadas:**
- **One-Hot Encoding:** Para géneros principales (top 5)
- **Label Encoding:** Para variables ordinales
- **Binary Encoding:** Para columnas con alta cardinalidad
- **Transformación logarítmica:** Corrección de sesgo en variables asimétricas
- **Standard Scaling:** Normalización de rating y runtime

**Visualizaciones generadas:**
- `boxplot_outliers.png`
- `histogramas_distribucion.png`
- `dispersion_rating_votes.png`
- `correlacion.png`

**Librerías:** pandas, numpy, matplotlib, seaborn, scikit-learn, category-encoders

---

### 4️⃣ Proyecto GTZAN: Clasificación de Géneros Musicales
**Archivo:** `gtzan/Pipeline.py`

Pipeline completo de análisis y preprocesamiento para clasificación de géneros musicales usando el dataset GTZAN.

**Análisis exploratorio:**
- Estadísticas descriptivas de características de audio
- Distribución de clases (balance del dataset)
- Análisis de "brillo" por género (spectral centroid)
- Distribución de tempo por género

**Procesamiento de características:**
- Eliminación de características altamente correlacionadas (>0.9)
- Visualización de espectrogramas Mel para géneros contrastantes (metal vs classical)
- Reducción de dimensionalidad con **t-SNE** para visualización 2D
- Label encoding de géneros musicales

**Visualizaciones generadas:**
- Distribución de géneros
- Brillo promedio por género
- Boxplots de tempo
- Matriz de correlación
- Espectrogramas Mel
- Visualización t-SNE

**Librerías:** pandas, numpy, matplotlib, seaborn, librosa, scikit-learn

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **Análisis de datos:** pandas, numpy
- **Visualización:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Audio processing:** librosa
- **Encoding:** category-encoders

---

## 📊 Habilidades Desarrolladas

✅ Análisis exploratorio de datos (EDA)  
✅ Limpieza y preprocesamiento de datos  
✅ Detección y manejo de outliers  
✅ Feature engineering  
✅ Técnicas de encoding (One-Hot, Label, Binary)  
✅ Normalización y escalado de datos  
✅ Reducción de dimensionalidad (PCA, t-SNE)  
✅ Visualización de datos  
✅ Procesamiento de señales de audio  
✅ Análisis de correlaciones

---

## 🚀 Cómo Ejecutar los Proyectos

Cada proyecto puede ejecutarse de forma independiente:

```bash
# Semana 2 - Iris
python sem02/ospina_nicolas_iris_analysis.py

# Semana 3 - Fintech
cd sem03/Fintech1000
python lab_fintech_sintetico_2025_1000.py

# Semana 4 - Películas
cd sem04
python EDA-NicolasOspina.py

# GTZAN - Música
cd gtzan
python Pipeline.py
```

**Nota:** Asegúrate de tener instaladas todas las dependencias necesarias antes de ejecutar.

---

## 📝 Notas

- Todos los proyectos incluyen visualizaciones guardadas como archivos `.png` o `.jpg`
- Los datasets procesados se exportan en formatos eficientes (CSV, Parquet)
- El código está documentado con comentarios explicativos en español

---

