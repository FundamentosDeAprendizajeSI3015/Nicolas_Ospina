"""
Titanic - Limpieza de datos, Balanceo de Clases y Regresión Logística

El dataset tiene desbalance en la variable target:
  ~62% No sobrevivió (0)  vs  ~38% Sobrevivió (1)

Estrategias de balanceo comparadas:
  1. class_weight='balanced'  → penaliza más los errores en la clase minoritaria
  2. Undersampling (RandomUnderSampler) → reduce la clase mayoritaria
  3. Oversampling (SMOTE)               → genera muestras sintéticas de la clase minoritaria

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ─────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────
train_df = pd.read_csv("titanic/train.csv")
test_df  = pd.read_csv("titanic/test.csv")

print("Shape train:", train_df.shape)
print(train_df.head())
print("\nValores nulos (train):\n", train_df.isnull().sum())

# Diagnóstico de desbalance
print("\n── Distribución de clases (target) ────")
print(train_df["Survived"].value_counts())
print(train_df["Survived"].value_counts(normalize=True).map("{:.1%}".format))

# ─────────────────────────────────────────
# 2. LIMPIEZA Y FEATURE ENGINEERING
# ─────────────────────────────────────────

def clean_and_engineer(df):
    df = df.copy()

    # --- Age: imputar con mediana agrupada por Pclass y Sex ---
    df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    # --- Embarked: imputar con moda ---
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # --- Fare: imputar con mediana (por si hay NaN en test) ---
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # --- Título extraído del nombre ---
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace(
        {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    )

    # --- Tamaño de familia ---
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # --- Cabin: indicador de si tiene cabina ---
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # --- Banda de fare ---
    df["FareBand"] = pd.qcut(df["Fare"], q=4, labels=False)

    # --- Banda de edad ---
    df["AgeBand"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100], labels=False)

    # --- Codificación de variables categóricas ---
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    df["Title"] = df["Title"].map(title_map).fillna(0)

    # --- Selección de features finales ---
    features = [
        "Pclass", "Sex", "AgeBand", "FareBand",
        "FamilySize", "IsAlone", "HasCabin",
        "Embarked", "Title"
    ]
    return df[features]


X = clean_and_engineer(train_df)
y = train_df["Survived"]
X_test_final = clean_and_engineer(test_df)

X = X.fillna(X.median())
X_test_final = X_test_final.fillna(X.median())  # usa la mediana del train para no hacer data leakage

print("\nFeatures shape:", X.shape)
print(X.dtypes)
print("\nNaN residuales en X:", X.isnull().sum().sum())  # debe ser 0

# ─────────────────────────────────────────
# 3. SPLIT Y ESCALADO
# ─────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test_final)

# ─────────────────────────────────────────
# 4. BALANCEO DE CLASES + ENTRENAMIENTO
# Tres estrategias comparadas
# ─────────────────────────────────────────

# Estrategia A: class_weight='balanced' (sin resampleo, ajusta pesos internamente)
# Estrategia B: Undersampling — reduce la clase mayoritaria al tamaño de la minoritaria
# Estrategia C: SMOTE — oversampling sintético de la clase minoritaria

X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X_train_sc, y_train)
X_smote, y_smote = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train_sc, y_train)

print("\n── Tamaños después del balanceo ────────")
print(f"  Original:     {dict(y_train.value_counts().sort_index())}")
print(f"  Undersampling:{dict(pd.Series(y_under).value_counts().sort_index())}")
print(f"  SMOTE:        {dict(pd.Series(y_smote).value_counts().sort_index())}")

strategies = {
    "Sin balanceo (baseline)":    (X_train_sc, y_train,  LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    "class_weight=balanced":      (X_train_sc, y_train,  LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight="balanced")),
    "Undersampling":              (X_under,    y_under,   LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    "SMOTE":                      (X_smote,    y_smote,   LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
}

results = {}
for name, (Xtr, ytr, clf) in strategies.items():
    clf.fit(Xtr, ytr)
    y_pred      = clf.predict(X_val_sc)
    y_pred_prob = clf.predict_proba(X_val_sc)[:, 1]
    results[name] = {"model": clf, "y_pred": y_pred, "y_prob": y_pred_prob}
    print(f"\n{'─'*50}")
    print(f"Estrategia: {name}")
    print(classification_report(y_val, y_pred, target_names=["No sobrevivió", "Sobrevivió"]))
    print(f"  ROC-AUC: {roc_auc_score(y_val, y_pred_prob):.4f}")

# ─────────────────────────────────────────
# 5. MODELO FINAL → SMOTE (mejor balance recall/precision)
# ─────────────────────────────────────────
best_name   = "SMOTE"
best_model  = results[best_name]["model"]
y_pred_best = results[best_name]["y_pred"]
y_prob_best = results[best_name]["y_prob"]

# ─────────────────────────────────────────
# 6. VISUALIZACIONES
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 6a. Distribución de clases original vs balanceada
counts = pd.DataFrame({
    "Original":     pd.Series(y_train).value_counts().sort_index(),
    "Undersampling": pd.Series(y_under).value_counts().sort_index(),
    "SMOTE":         pd.Series(y_smote).value_counts().sort_index(),
}, index=["No sobrevivió", "Sobrevivió"])
counts.plot(kind="bar", ax=axes[0], colormap="Set2", rot=0)
axes[0].set_title("Distribución de Clases\npor Estrategia de Balanceo")
axes[0].set_ylabel("Número de muestras")
axes[0].legend(fontsize=8)

# 6b. Matriz de confusión (modelo SMOTE)
cm = confusion_matrix(y_val, y_pred_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["No sobrevivió", "Sobrevivió"],
            yticklabels=["No sobrevivió", "Sobrevivió"])
axes[1].set_title(f"Matriz de Confusión\n({best_name})")
axes[1].set_ylabel("Real")
axes[1].set_xlabel("Predicho")

# 6c. Curvas ROC de todas las estrategias
colors = ["gray", "steelblue", "darkorange", "green"]
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_val, res["y_prob"])
    auc = roc_auc_score(y_val, res["y_prob"])
    axes[2].plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", lw=1)
axes[2].set_xlabel("Tasa de Falsos Positivos")
axes[2].set_ylabel("Tasa de Verdaderos Positivos")
axes[2].set_title("Curvas ROC — Comparativa")
axes[2].legend(fontsize=7)

plt.tight_layout()
plt.savefig("titanic_results.png", dpi=150)
plt.show()

# Coeficientes del modelo final
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coeficiente": best_model.coef_[0]
}).sort_values("Coeficiente", ascending=False)

print("\n── Coeficientes del Modelo Final (SMOTE) ──")
print(coef_df.to_string(index=False))

# ─────────────────────────────────────────
# 7. PREDICCIONES PARA SUBMISSION
# ─────────────────────────────────────────
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": best_model.predict(X_test_sc)
})
submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv generado correctamente")