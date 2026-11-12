"""Flujo de entrenamiento coordinado con Prefect, independiente de Jupyter.

Etapas: adquisición → validación/particiones → baseline → tuning por familia (validación) →
selección champion/challenger → reajuste con entrenamiento+validación y evaluación en prueba →
registro en Unity Catalog con aliases → métricas en `reports/metricas.csv`.

Uso: `uv run python src/train_flow.py`
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from optuna.samplers import TPESampler
from prefect import flow, get_run_logger, task

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos  # noqa: E402
import entrenamiento as ent  # noqa: E402

RAIZ = datos.RAIZ
DIR_ARTIFACTS = RAIZ / "artifacts"
RUTA_METRICAS = RAIZ / "reports" / "metricas.csv"
RUTA_SELECCION = DIR_ARTIFACTS / "seleccion.json"
RUTA_PRED_TEST = DIR_ARTIFACTS / "predicciones_test.parquet"

EXPERIMENTO = "/Shared/pcd-2026-examen-sms-remih06"
NOMBRE_MODELO = os.getenv("MODEL_NAME", "workspace.default.pcd_2026_examen_sms_remih06")
ALIASES = ("champion", "challenger")
COMUN = {"course": "proyecto-ciencia-datos", "term": "otono-2026", "task": "examen-final"}

# Texto inventado (sección 5: nunca se registran mensajes del corpus).
EJEMPLO_ENTRADA = pd.DataFrame({ent.COLUMNA_TEXTO: ["Can we meet after class?"]})


def contexto_datos(resumen: dict) -> dict:
    """Parámetros de procedencia que acompañan a cada run (sin mensajes del corpus)."""
    tamanos = resumen["particiones"]["tamanos"]
    return {
        "data_source": datos.URL_ZIP,
        "data_downloaded_utc": resumen["descarga"]["fecha_utc"],
        "data_sha256": resumen["descarga"]["sha256_corpus"],
        "data_clean_rows": resumen["depuracion"]["registros_limpios"],
        "split_seed": resumen["particiones"]["semilla"],
        "split_strategy": "60/20/20 estratificado, sin claves compartidas",
        "n_train": tamanos["train"],
        "n_validation": tamanos["validation"],
        "n_test": tamanos["test"],
        "selection_metric": ent.METRICA_PRINCIPAL,
        "tiebreak_metric": ent.METRICA_DESEMPATE,
    }


def con_prefijo(prefijo: str, m: dict) -> dict:
    return {f"{prefijo}_{k}": v for k, v in m.items()}


# --- Datos -------------------------------------------------------------------

@task(retries=3, retry_delay_seconds=[5, 15, 30])
def adquirir() -> Path:
    """Descarga sólo si no hay copia local; un fallo de red se reintenta."""
    ruta = datos.descargar()
    get_run_logger().info("Corpus disponible en %s", ruta.relative_to(RAIZ))
    return ruta


@task
def preparar_particiones(_ruta_corpus: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Valida, depura y separa. ContratoInvalido detiene el flujo: reintentar no lo arregla."""
    resumen = datos.preparar()
    particiones = datos.cargar_particiones()
    get_run_logger().info(
        "Limpios: %d | tamaños: %s | claves compartidas: %s",
        resumen["depuracion"]["registros_limpios"],
        resumen["particiones"]["tamanos"],
        resumen["particiones"]["claves_compartidas"],
    )
    return resumen, particiones


# --- Selección (sólo entrenamiento y validación) ----------------------------

@task
def evaluar_baseline(train: pd.DataFrame, validation: pd.DataFrame, contexto: dict) -> dict:
    inicio = perf_counter()
    _, m, _ = ent.ajustar_y_evaluar("baseline", {}, train, validation)
    duracion = perf_counter() - inicio
    with mlflow.start_run(run_name="baseline-validacion") as run:
        mlflow.log_params({**contexto, "model_family": "baseline", "strategy": "most_frequent"})
        mlflow.log_metrics({**con_prefijo("val", m), "duration_seconds": duracion})
        mlflow.set_tags({**COMUN, "role": "baseline", "model_family": "baseline"})
    return {"familia": "baseline", "params": {"strategy": "most_frequent"}, "val": m, "run_id": run.info.run_id}


@task(task_run_name="tuning-{familia}")
def tunear_familia(familia: str, train: pd.DataFrame, validation: pd.DataFrame, contexto: dict) -> dict:
    """Optuna TPE sobre validación. Cada trial ajusta un pipeline nuevo sólo con entrenamiento."""
    logger = get_run_logger()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    espacio = ent.ESPACIOS[familia]
    resultados: dict[int, dict] = {}

    with mlflow.start_run(run_name=f"tuning-{familia}") as padre:
        mlflow.log_params({
            **contexto, "model_family": familia, "optimizer": "optuna-tpe",
            "sampler_seed": ent.SEMILLA, "n_trials": ent.N_TRIALS,
        })
        mlflow.set_tags({**COMUN, "role": "tuning-study", "model_family": familia})

        def objetivo(trial: optuna.trial.Trial) -> float:
            params = espacio(trial)
            inicio = perf_counter()
            _, m, _ = ent.ajustar_y_evaluar(familia, params, train, validation)
            duracion = perf_counter() - inicio
            with mlflow.start_run(run_name=f"{familia}-trial-{trial.number:02d}", nested=True) as hijo:
                mlflow.log_params({**params, "model_family": familia, "trial_number": trial.number})
                mlflow.log_metrics({**con_prefijo("val", m), "duration_seconds": duracion})
                mlflow.set_tags({**COMUN, "role": "tuning-trial", "model_family": familia})
            resultados[trial.number] = {"params": params, "val": m, "run_id": hijo.info.run_id}
            return m[ent.METRICA_PRINCIPAL]

        estudio = optuna.create_study(direction="maximize", sampler=TPESampler(seed=ent.SEMILLA))
        estudio.optimize(objetivo, n_trials=ent.N_TRIALS)

        # Optuna maximiza sólo F1; el desempate por precision se aplica aquí sobre todos los trials.
        numero, mejor = max(resultados.items(), key=lambda kv: ent.clave_seleccion(kv[1]["val"]))
        mlflow.log_params({"best_trial_number": numero, **{f"best_{k}": v for k, v in mejor["params"].items()}})
        mlflow.log_metrics(con_prefijo("best_val", mejor["val"]))

    logger.info("%s: trial %d, F1 val %.4f, precision val %.4f, %s", familia, numero,
                mejor["val"]["f1_spam"], mejor["val"]["precision_spam"], mejor["params"])
    return {"familia": familia, "trial": numero, "study_run_id": padre.info.run_id, **mejor}


@task
def seleccionar(finalistas: list[dict]) -> list[dict]:
    """Champion = mejor finalista en validación (F1 spam, desempate precision). Se cierra aquí."""
    ranking = sorted(finalistas, key=lambda f: ent.clave_seleccion(f["val"]), reverse=True)
    for finalista, alias in zip(ranking, ALIASES):
        finalista["alias"] = alias
    DIR_ARTIFACTS.mkdir(exist_ok=True)
    RUTA_SELECCION.write_text(json.dumps(
        [{k: f[k] for k in ("alias", "familia", "params", "val", "run_id", "trial")} for f in ranking],
        indent=2, ensure_ascii=False,
    ) + "\n", encoding="utf-8")
    get_run_logger().info("Selección cerrada: %s", [(f["alias"], f["familia"]) for f in ranking])
    return ranking


# --- Evaluación final (prueba, sin retuning) y registro ---------------------

@task(task_run_name="prueba-{candidato[familia]}")
def evaluar_en_prueba(candidato: dict, train_val: pd.DataFrame, test: pd.DataFrame, contexto: dict,
                      registrar: bool) -> dict:
    """Reajusta con entrenamiento+validación los parámetros ya elegidos y mide una vez en prueba."""
    familia = candidato["familia"]
    params = {} if familia == "baseline" else candidato["params"]
    inicio = perf_counter()
    pipeline, m, predichas = ent.ajustar_y_evaluar(familia, params, train_val, test)
    duracion = perf_counter() - inicio
    version = None
    with mlflow.start_run(run_name=f"final-{familia}") as run:
        mlflow.log_params({**contexto, **candidato["params"], "model_family": familia,
                           "trained_on": "train+validation", "source_selection_run": candidato["run_id"]})
        mlflow.log_metrics({**con_prefijo("val", candidato["val"]), **con_prefijo("test", m),
                            "duration_seconds": duracion})
        mlflow.set_tags({**COMUN, "role": "final-evaluation", "model_family": familia,
                         "selection_alias": candidato.get("alias", "baseline")})
        mlflow.log_dict({"labels": ent.ETIQUETAS, "matrix": ent.matriz_confusion(test["label"], predichas)},
                        "confusion_matrix_test.json")
        if registrar:
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=EJEMPLO_ENTRADA,
                signature=infer_signature(EJEMPLO_ENTRADA, pipeline.predict(EJEMPLO_ENTRADA)),
                registered_model_name=NOMBRE_MODELO,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
            )
            version = str(info.registered_model_version)
    get_run_logger().info("%s: F1 prueba %.4f, versión %s", familia, m["f1_spam"], version)
    return {**candidato, "test": m, "final_run_id": run.info.run_id, "version": version,
            "predichas": list(predichas), "matriz": ent.matriz_confusion(test["label"], predichas)}


@task(retries=2, retry_delay_seconds=10)
def asignar_aliases(finales: list[dict]) -> None:
    cliente = MlflowClient()
    for f in finales:
        cliente.set_registered_model_alias(NOMBRE_MODELO, f["alias"], f["version"])
        etiquetas = {
            "model_family": f["familia"],
            "selection_role": f["alias"],
            "val_f1_spam": f"{f['val']['f1_spam']:.4f}",
            "test_f1_spam": f"{f['test']['f1_spam']:.4f}",
            "trained_on": "train+validation",
            "selection_run_id": f["run_id"],
        }
        for clave, valor in etiquetas.items():
            cliente.set_model_version_tag(NOMBRE_MODELO, f["version"], clave, valor)
        get_run_logger().info("%s → %s v%s", f["alias"], f["familia"], f["version"])


@task
def guardar_resultados(baseline: dict, finalistas: list[dict], finales: list[dict], test: pd.DataFrame) -> Path:
    filas = []
    for c in [baseline, *finalistas]:
        filas.append({"conjunto": "validation", "familia": c["familia"], "alias": c.get("alias", ""),
                      "configuracion": json.dumps(c["params"], sort_keys=True), **c["val"],
                      "experimento": EXPERIMENTO, "run_id": c["run_id"], "version_modelo": ""})
    for f in finales:
        filas.append({"conjunto": "test", "familia": f["familia"], "alias": f.get("alias", ""),
                      "configuracion": json.dumps(f["params"], sort_keys=True), **f["test"],
                      "experimento": EXPERIMENTO, "run_id": f["final_run_id"], "version_modelo": f["version"] or ""})
    tabla = pd.DataFrame(filas)
    tabla[["accuracy", "precision_spam", "recall_spam", "f1_spam"]] = tabla[
        ["accuracy", "precision_spam", "recall_spam", "f1_spam"]].round(4)
    RUTA_METRICAS.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_METRICAS, index=False)

    # Predicciones de prueba para el análisis de errores del informe. Contienen texto del
    # corpus, por eso viven en artifacts/ (ignorado por Git) y nunca se suben a MLflow.
    pred = test[["text", "label"]].copy()
    for f in finales:
        pred[f"pred_{f['familia']}"] = f["predichas"]
    pred.to_parquet(RUTA_PRED_TEST, index=False)
    return RUTA_METRICAS


# --- Flow --------------------------------------------------------------------

@flow(name="nimbo-sms-entrenamiento", log_prints=True)
def entrenar() -> list[dict]:
    load_dotenv(RAIZ / ".env", override=True)
    if not os.getenv("DATABRICKS_HOST") or not os.getenv("DATABRICKS_TOKEN"):
        raise ValueError("Faltan DATABRICKS_HOST o DATABRICKS_TOKEN en .env (ver .env.example)")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(EXPERIMENTO)

    ruta = adquirir()
    resumen, p = preparar_particiones(ruta)
    contexto = contexto_datos(resumen)

    baseline = evaluar_baseline(p["train"], p["validation"], contexto)
    finalistas = [tunear_familia(f, p["train"], p["validation"], contexto) for f in ent.FAMILIAS]
    ranking = seleccionar(finalistas)

    train_val = pd.concat([p["train"], p["validation"]], ignore_index=True)
    baseline_final = evaluar_en_prueba(baseline, train_val, p["test"], contexto, registrar=False)
    finales = [evaluar_en_prueba(c, train_val, p["test"], contexto, registrar=True) for c in ranking]
    asignar_aliases(finales)
    guardar_resultados(baseline, ranking, [baseline_final, *finales], p["test"])

    resumen_final = [{"alias": f["alias"], "familia": f["familia"], "version": f["version"],
                      "val_f1": round(f["val"]["f1_spam"], 4), "test_f1": round(f["test"]["f1_spam"], 4)}
                     for f in finales]
    print(json.dumps(resumen_final, indent=2))
    return resumen_final


if __name__ == "__main__":
    entrenar()
