"""API de Nimbo SMS: sirve la versión del modelo que tiene el alias configurado al iniciar.

Al arrancar resuelve `MODEL_ALIAS` (champion) a una versión concreta del registro y la carga
una sola vez; esa versión se conserva mientras corre el proceso. Un cambio de alias se adopta
al reiniciar. No reentrena ni aplica reglas propias: el texto crudo entra al pipeline registrado.

Uso local: `uv run uvicorn api.main:app --host 127.0.0.1 --port 8060`
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field, field_validator

load_dotenv()  # en local lee .env; en Docker las variables llegan por env_file
logger = logging.getLogger("uvicorn.error")

MODEL_NAME = os.getenv("MODEL_NAME", "workspace.default.pcd_2026_examen_sms_remih06")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MAX_CARACTERES = 2000

estado: dict = {"modelo": None, "version": None, "error": None}


def cargar_modelo() -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS).version
    # Se carga por número de versión, no por alias, para fijar exactamente lo que se resolvió.
    estado["modelo"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
    estado["version"] = str(version)
    logger.info("Modelo %s@%s → versión %s cargado", MODEL_NAME, MODEL_ALIAS, version)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        cargar_modelo()
    except Exception as exc:  # la API arranca, pero /health responde 503 hasta reiniciar con éxito
        estado["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("No se pudo cargar el modelo: %s", estado["error"])
    yield


app = FastAPI(
    title="Nimbo SMS API",
    description="Clasificación estimada ham/spam de SMS en inglés. Apoya la decisión; no bloquea mensajes.",
    version="1.0.0",
    lifespan=lifespan,
)


class Solicitud(BaseModel):
    # extra="forbid" rechaza campos adicionales; strict=True rechaza {"text": 7} sin convertirlo.
    model_config = ConfigDict(extra="forbid", strict=True)

    text: Annotated[str, Field(min_length=1, max_length=MAX_CARACTERES, examples=["Can we meet after class?"])]

    @field_validator("text")
    @classmethod
    def no_solo_espacios(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("text no puede contener sólo espacios")
        return valor  # se devuelve sin modificar: el modelo recibe el texto crudo


class Prediccion(BaseModel):
    label: Literal["ham", "spam"]
    model_name: str
    model_version: str


class Salud(BaseModel):
    status: Literal["ok"]
    model_name: str
    model_alias: str
    model_version: str


@app.get("/health", response_model=Salud, responses={503: {"description": "Modelo no cargado"}})
def health():
    if estado["modelo"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_name": MODEL_NAME, "model_alias": MODEL_ALIAS,
                     "detail": estado["error"] or "cargando"},
        )
    return Salud(status="ok", model_name=MODEL_NAME, model_alias=MODEL_ALIAS, model_version=estado["version"])


@app.post("/predict", response_model=Prediccion, responses={503: {"description": "Modelo no cargado"}})
def predict(solicitud: Solicitud) -> Prediccion:
    if estado["modelo"] is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    etiqueta = estado["modelo"].predict(pd.DataFrame({"text": [solicitud.text]}))[0]
    return Prediccion(label=str(etiqueta), model_name=MODEL_NAME, model_version=estado["version"])
