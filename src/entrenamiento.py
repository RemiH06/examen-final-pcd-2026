"""Representación, familias de modelos, espacios de búsqueda, métricas y selección.

Funciones puras (sin Prefect ni MLflow) para que el flujo, el notebook y las pruebas de
paridad usen exactamente la misma construcción del modelo.

Representación: TF-IDF dentro de un `ColumnTransformer` sobre la columna `text`. Así el
modelo completo (vocabulario + clasificador) es un solo objeto de scikit-learn que recibe
el mismo DataFrame `{"text": ...}` en entrenamiento, evaluación y en la API, y el
vocabulario se aprende en cada `fit`, nunca antes de separar los datos.
"""

from __future__ import annotations

import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SEMILLA = 42
ETIQUETAS = ["ham", "spam"]
POSITIVA = "spam"
COLUMNA_TEXTO = "text"

# Definidas ANTES del tuning (sección 7.4). F1 de spam equilibra spam no detectado (FN)
# y mensajes legítimos marcados (FP); con 12 % de spam, accuracy premia predecir siempre ham.
# Desempate: precision de spam, porque un FP oculta un mensaje legítimo a la persona.
METRICA_PRINCIPAL = "f1_spam"
METRICA_DESEMPATE = "precision_spam"

FAMILIAS = ("logistic_regression", "multinomial_nb")
N_TRIALS = 20  # por familia; cada ajuste tarda < 1 s, así que 20 cubre el espacio sin costo relevante


def construir_pipeline(familia: str, params: dict) -> Pipeline:
    """Arma vectorizador + clasificador a partir de un diccionario plano de parámetros."""
    if familia == "baseline":
        # El baseline ignora el texto, pero conserva la misma interfaz de entrada.
        tfidf = TfidfVectorizer()
        clasificador = DummyClassifier(strategy="most_frequent")
    else:
        tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, params["ngram_max"]),
            min_df=params["min_df"],
            sublinear_tf=params["sublinear_tf"],
        )
        if familia == "logistic_regression":
            clasificador = LogisticRegression(
                C=params["C"],
                class_weight=params["class_weight"],
                max_iter=2000,
                random_state=SEMILLA,
            )
        elif familia == "multinomial_nb":
            clasificador = MultinomialNB(alpha=params["alpha"], fit_prior=params["fit_prior"])
        else:
            raise ValueError(f"Familia no reconocida: {familia}")
    representacion = ColumnTransformer([("tfidf", tfidf, COLUMNA_TEXTO)])
    return Pipeline([("representacion", representacion), ("clasificador", clasificador)])


def espacio_tfidf(trial: optuna.trial.Trial) -> dict:
    return {
        "ngram_max": trial.suggest_int("ngram_max", 1, 2),
        "min_df": trial.suggest_int("min_df", 1, 3),
        "sublinear_tf": trial.suggest_categorical("sublinear_tf", [False, True]),
    }


def espacio_logistic_regression(trial: optuna.trial.Trial) -> dict:
    # C de 0.1 a 100 en escala log: de regularización fuerte a casi nula. class_weight
    # "balanced" compensa el 12 % de spam dando más peso a sus errores.
    return {
        **espacio_tfidf(trial),
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def espacio_multinomial_nb(trial: optuna.trial.Trial) -> dict:
    # alpha de 0.001 a 1 en escala log: de casi sin suavizado al Laplace clásico.
    # fit_prior=False usa priors uniformes, otra forma de atender el desbalance.
    return {
        **espacio_tfidf(trial),
        "alpha": trial.suggest_float("alpha", 1e-3, 1.0, log=True),
        "fit_prior": trial.suggest_categorical("fit_prior", [True, False]),
    }


ESPACIOS = {
    "logistic_regression": espacio_logistic_regression,
    "multinomial_nb": espacio_multinomial_nb,
}


def X(df: pd.DataFrame) -> pd.DataFrame:
    return df[[COLUMNA_TEXTO]]


def metricas(reales, predichas) -> dict:
    return {
        "accuracy": accuracy_score(reales, predichas),
        "precision_spam": precision_score(reales, predichas, pos_label=POSITIVA, zero_division=0),
        "recall_spam": recall_score(reales, predichas, pos_label=POSITIVA, zero_division=0),
        "f1_spam": f1_score(reales, predichas, pos_label=POSITIVA, zero_division=0),
    }


def matriz_confusion(reales, predichas) -> list[list[int]]:
    """Filas reales, columnas predichas, orden ham, spam."""
    return confusion_matrix(reales, predichas, labels=ETIQUETAS).tolist()


def clave_seleccion(m: dict) -> tuple[float, float]:
    return (m[METRICA_PRINCIPAL], m[METRICA_DESEMPATE])


def ajustar_y_evaluar(familia: str, params: dict, entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame):
    """Crea un pipeline nuevo, lo ajusta sólo con `entrenamiento` y mide sobre `evaluacion`."""
    pipeline = construir_pipeline(familia, params)
    pipeline.fit(X(entrenamiento), entrenamiento["label"])
    predichas = pipeline.predict(X(evaluacion))
    return pipeline, metricas(evaluacion["label"], predichas), predichas
