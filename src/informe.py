"""Funciones del informe: leen resultados ya guardados por el flujo y producen tablas y gráficas.

No entrenan, no ajustan vocabulario ni registran modelos. Ninguna función devuelve texto de
mensajes del corpus: las tablas de errores sólo muestran rasgos numéricos (sección 5).
"""

from __future__ import annotations

import json
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import datos

RAIZ = datos.RAIZ
COLORES = {"ham": "#4C78A8", "spam": "#E45756"}


# --- Lectura de resultados guardados -----------------------------------------

def resumen_datos() -> dict:
    return json.loads((RAIZ / "reports" / "datos-resumen.json").read_text(encoding="utf-8"))


def metricas() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "reports" / "metricas.csv")


def seleccion() -> list[dict]:
    return json.loads((RAIZ / "artifacts" / "seleccion.json").read_text(encoding="utf-8"))


def predicciones_test() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "artifacts" / "predicciones_test.parquet")


# --- Calidad y particiones ---------------------------------------------------

def tabla_depuracion(resumen: dict) -> pd.DataFrame:
    d = resumen["depuracion"]
    filas = [
        ("Registros leídos", d["registros_leidos"]),
        ("Textos vacíos eliminados", d["textos_vacios_eliminados"]),
        ("Claves contradictorias (ham y spam)", d["claves_contradictorias"]),
        ("Duplicados eliminados", d["duplicados_eliminados"]),
        ("  de ellos ham", d["duplicados_por_etiqueta"].get("ham", 0)),
        ("  de ellos spam", d["duplicados_por_etiqueta"].get("spam", 0)),
        ("Registros limpios", d["registros_limpios"]),
    ]
    return pd.DataFrame(filas, columns=["Paso", "Conteo"])


def tabla_particiones(particiones: dict[str, pd.DataFrame]) -> pd.DataFrame:
    filas = []
    for nombre, df in particiones.items():
        conteo = df["label"].value_counts()
        filas.append({"particion": nombre, "mensajes": len(df), "ham": conteo.get("ham", 0),
                      "spam": conteo.get("spam", 0), "proporcion_spam": round(conteo.get("spam", 0) / len(df), 4)})
    return pd.DataFrame(filas)


def claves_compartidas(particiones: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Recalcula la verificación de fuga a partir de las particiones guardadas."""
    claves = {n: set(datos.clave(df["text"])) for n, df in particiones.items()}
    nombres = list(claves)
    return pd.DataFrame([
        {"par": f"{a} ∩ {b}", "mensajes_equivalentes_compartidos": len(claves[a] & claves[b])}
        for i, a in enumerate(nombres) for b in nombres[i + 1:]
    ])


# --- EDA (sólo entrenamiento) ------------------------------------------------

def rasgos(textos: pd.Series) -> pd.DataFrame:
    """Rasgos numéricos para describir mensajes sin mostrarlos."""
    return pd.DataFrame({
        "caracteres": textos.str.len(),
        "palabras": textos.str.split().str.len(),
        "digitos": textos.str.count(r"\d"),
        "tiene_numero_largo": textos.str.contains(r"\d{5,}"),
        "tiene_moneda": textos.str.contains(r"[£$€]|\bcash\b|\bprize\b", flags=re.IGNORECASE),
        "tiene_url": textos.str.contains(r"http|www\.|\.com", flags=re.IGNORECASE),
        "proporcion_mayusculas": textos.apply(lambda t: sum(c.isupper() for c in t) / max(len(t), 1)),
    }, index=textos.index)


def fig_clases(train: pd.DataFrame) -> go.Figure:
    conteo = train["label"].value_counts().reindex(["ham", "spam"]).reset_index()
    conteo.columns = ["etiqueta", "mensajes"]
    conteo["porcentaje"] = (conteo["mensajes"] / conteo["mensajes"].sum() * 100).round(1)
    fig = px.bar(conteo, x="etiqueta", y="mensajes", color="etiqueta", color_discrete_map=COLORES,
                 text=conteo["porcentaje"].astype(str) + " %",
                 title="Distribución de clases en entrenamiento")
    fig.update_layout(showlegend=False, template="plotly_white")
    return fig


def fig_longitud(train: pd.DataFrame) -> go.Figure:
    df = pd.DataFrame({"etiqueta": train["label"], "caracteres": train["text"].str.len()})
    fig = px.histogram(df, x="caracteres", color="etiqueta", color_discrete_map=COLORES, barmode="overlay",
                       histnorm="percent", nbins=60, opacity=0.65, marginal="box",
                       title="Longitud de los mensajes por clase (entrenamiento, % dentro de cada clase)")
    fig.update_layout(template="plotly_white", yaxis_title="% de mensajes de la clase")
    return fig


def tabla_rasgos_por_clase(train: pd.DataFrame) -> pd.DataFrame:
    r = rasgos(train["text"]).assign(etiqueta=train["label"])
    return r.groupby("etiqueta").agg(
        mediana_caracteres=("caracteres", "median"),
        mediana_palabras=("palabras", "median"),
        mediana_digitos=("digitos", "median"),
        con_numero_largo=("tiene_numero_largo", "mean"),
        con_moneda_o_premio=("tiene_moneda", "mean"),
        con_url=("tiene_url", "mean"),
        mayusculas_media=("proporcion_mayusculas", "mean"),
    ).round(3)


# --- Modelos y evaluación ----------------------------------------------------

def tabla_comparacion(m: pd.DataFrame) -> pd.DataFrame:
    cols = ["conjunto", "familia", "alias", "accuracy", "precision_spam", "recall_spam", "f1_spam", "version_modelo"]
    return m[cols].fillna("").sort_values(["conjunto", "f1_spam"], ascending=[False, False]).reset_index(drop=True)


def fig_comparacion(m: pd.DataFrame) -> go.Figure:
    largo = m.melt(id_vars=["conjunto", "familia"], value_vars=["accuracy", "precision_spam", "recall_spam", "f1_spam"],
                   var_name="metrica", value_name="valor")
    fig = px.bar(largo, x="metrica", y="valor", color="familia", barmode="group", facet_col="conjunto",
                 category_orders={"conjunto": ["validation", "test"]}, text_auto=".3f",
                 title="Baseline y finalistas: validación (selección) y prueba (evaluación final)")
    fig.update_layout(template="plotly_white", yaxis_range=[0, 1.05])
    return fig


def matriz_confusion(pred: pd.DataFrame, familia: str) -> pd.DataFrame:
    tabla = pd.crosstab(pred["label"], pred[f"pred_{familia}"]).reindex(
        index=["ham", "spam"], columns=["ham", "spam"], fill_value=0)
    tabla.index.name, tabla.columns.name = "real", "predicha"
    return tabla


def fig_matriz(tabla: pd.DataFrame, familia: str) -> go.Figure:
    fig = px.imshow(tabla.values, x=["ham", "spam"], y=["ham", "spam"], text_auto=True,
                    color_continuous_scale="Blues", labels={"x": "Predicha", "y": "Real", "color": "Mensajes"},
                    title=f"Matriz de confusión del champion ({familia}) en prueba")
    fig.update_layout(template="plotly_white")
    return fig


def errores(pred: pd.DataFrame, familia: str, maximo: int = 3) -> pd.DataFrame:
    """Hasta `maximo` FP y FN con rasgos numéricos y un identificador; nunca el texto."""
    col = f"pred_{familia}"
    fp = pred[(pred["label"] == "ham") & (pred[col] == "spam")].head(maximo).assign(tipo="FP (ham → spam)")
    fn = pred[(pred["label"] == "spam") & (pred[col] == "ham")].head(maximo).assign(tipo="FN (spam → ham)")
    sel = pd.concat([fp, fn])
    return pd.concat([sel[["tipo"]], rasgos(sel["text"])], axis=1).rename_axis("id_prueba").reset_index()


def conteo_errores(pred: pd.DataFrame, familia: str) -> dict:
    col = f"pred_{familia}"
    return {"FP": int(((pred["label"] == "ham") & (pred[col] == "spam")).sum()),
            "FN": int(((pred["label"] == "spam") & (pred[col] == "ham")).sum())}


# --- Registro ----------------------------------------------------------------

def versiones_registradas(nombre_modelo: str) -> pd.DataFrame:
    """Consulta de sólo lectura al registro de Unity Catalog (requiere .env con Databricks)."""
    import mlflow
    from dotenv import load_dotenv
    from mlflow import MlflowClient

    load_dotenv(RAIZ / ".env", override=True)
    mlflow.set_tracking_uri("databricks")  # sin esto MLflow 3 crea un mlflow.db local
    mlflow.set_registry_uri("databricks-uc")
    cliente = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")
    filas = []
    for v in cliente.search_model_versions(f"name='{nombre_modelo}'"):
        detalle = cliente.get_model_version(nombre_modelo, v.version)
        filas.append({"version": int(v.version), "aliases": ", ".join(detalle.aliases),
                      "familia": detalle.tags.get("model_family"), "val_f1_spam": detalle.tags.get("val_f1_spam"),
                      "test_f1_spam": detalle.tags.get("test_f1_spam"), "run_id": v.run_id, "estado": v.status})
    return pd.DataFrame(filas).sort_values("version").reset_index(drop=True)
