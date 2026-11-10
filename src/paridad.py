"""Comprueba que la API y el modelo registrado den la misma etiqueta con tres textos inventados.

Requiere la API corriendo. Uso: `uv run python src/paridad.py [--api http://127.0.0.1:8060]`
Escribe `reports/paridad.json`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import requests
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[1]

# Inventados (sección 5): uno claramente legítimo, uno con el estilo típico de spam y uno ambiguo.
TEXTOS = [
    "Can we meet after class to review the project notes?",
    "URGENT! You have won a 500 cash prize. Text WIN to 80888 now to claim, T&Cs apply.",
    "Your package is waiting, reply to confirm the delivery time for tomorrow.",
]


def main(api: str) -> None:
    load_dotenv(RAIZ / ".env", override=True)
    salud = requests.get(f"{api}/health", timeout=10).json()
    nombre, version = salud["model_name"], salud["model_version"]

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}/{version}")
    directas = modelo.predict(pd.DataFrame({"text": TEXTOS}))

    filas = []
    for texto, directa in zip(TEXTOS, directas):
        respuesta = requests.post(f"{api}/predict", json={"text": texto}, timeout=10)
        cuerpo = respuesta.json()
        filas.append({"texto": texto, "modelo": str(directa), "api": cuerpo["label"],
                      "http": respuesta.status_code, "version_api": cuerpo["model_version"],
                      "coinciden": str(directa) == cuerpo["label"]})

    resultado = {"modelo": nombre, "version": version, "todas_coinciden": all(f["coinciden"] for f in filas),
                 "casos": filas}
    (RAIZ / "reports" / "paridad.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\nVersión {version}: todas coinciden = {resultado['todas_coinciden']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default=os.getenv("API_URL", "http://127.0.0.1:8060"))
    main(parser.parse_args().api)
