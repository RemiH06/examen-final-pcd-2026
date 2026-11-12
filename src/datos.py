"""Adquisición, validación, depuración y particiones del SMS Spam Collection.

Responsabilidad: dejar en `data/processed/` tres particiones sin mensajes equivalentes
compartidos y en `reports/datos-resumen.json` los conteos de cada paso (sin mensajes).
Nada aquí aprende vocabulario ni elige modelos: eso ocurre después, sólo con entrenamiento.

Uso independiente: `uv run python src/datos.py`
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
DIR_CRUDOS = RAIZ / "data" / "raw"
DIR_PROCESADOS = RAIZ / "data" / "processed"
RUTA_RESUMEN = RAIZ / "reports" / "datos-resumen.json"

URL_ZIP = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
NOMBRE_ZIP = "sms+spam+collection.zip"
NOMBRE_CORPUS = "SMSSpamCollection"
FUENTE = {
    "nombre": "SMS Spam Collection (UCI)",
    "autores": "Almeida e Hidalgo (2011)",
    "doi": "10.24432/C5CC84",
    "licencia": "CC BY 4.0",
    "url": URL_ZIP,
}

ETIQUETAS = ("ham", "spam")
SEMILLA = 42
PROPORCIONES = {"train": 0.6, "validation": 0.2, "test": 0.2}
PARTICIONES = tuple(PROPORCIONES)


class ContratoInvalido(ValueError):
    """Los datos no cumplen el formato esperado; reintentar la descarga no lo arregla."""


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def descargar(destino: Path = DIR_CRUDOS) -> Path:
    """Descarga el ZIP oficial si no existe y extrae el corpus. Reutiliza la copia local."""
    destino.mkdir(parents=True, exist_ok=True)
    ruta_zip = destino / NOMBRE_ZIP
    ruta_corpus = destino / NOMBRE_CORPUS
    ruta_meta = destino / "descarga.json"
    if not ruta_zip.exists():
        temporal = ruta_zip.with_suffix(".zip.part")
        urlretrieve(URL_ZIP, temporal)
        temporal.replace(ruta_zip)  # sólo queda el ZIP si la descarga terminó
        ruta_meta.write_text(
            json.dumps({"url": URL_ZIP, "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}),
            encoding="utf-8",
        )
    if not ruta_corpus.exists():
        with zipfile.ZipFile(ruta_zip) as archivo:
            if NOMBRE_CORPUS not in archivo.namelist():
                raise ContratoInvalido(f"{NOMBRE_ZIP} no contiene {NOMBRE_CORPUS}")
            archivo.extract(NOMBRE_CORPUS, destino)
    return ruta_corpus


def cargar(ruta: Path) -> pd.DataFrame:
    """Lee el archivo sin encabezado separando sólo el primer tabulador de cada línea."""
    filas = []
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), start=1):
        if not linea:
            continue
        if "\t" not in linea:
            raise ContratoInvalido(f"Línea {numero} sin tabulador")
        etiqueta, texto = linea.split("\t", 1)
        filas.append((etiqueta, texto))
    return pd.DataFrame(filas, columns=["label", "text"])


def clave(texto: pd.Series) -> pd.Series:
    """Criterio de equivalencia entre mensajes: mismo texto ignorando bordes y mayúsculas."""
    return texto.str.strip().str.lower()


def depurar(crudo: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Valida etiquetas y vacíos, excluye claves contradictorias y conserva la primera aparición."""
    etiquetas_raras = sorted(set(crudo["label"]) - set(ETIQUETAS))
    if etiquetas_raras:
        raise ContratoInvalido(f"Etiquetas fuera de {ETIQUETAS}: {etiquetas_raras}")

    datos = crudo.assign(key=clave(crudo["text"]))
    vacios = datos["key"].eq("")
    datos = datos.loc[~vacios]

    etiquetas_por_clave = datos.groupby("key")["label"].nunique()
    contradictorias = etiquetas_por_clave[etiquetas_por_clave > 1].index
    filas_contradictorias = datos["key"].isin(contradictorias)
    datos = datos.loc[~filas_contradictorias]

    duplicados = datos.duplicated("key", keep="first")
    limpio = datos.loc[~duplicados].reset_index(drop=True)

    conteos = {
        "registros_leidos": len(crudo),
        "por_etiqueta_leidos": crudo["label"].value_counts().to_dict(),
        "textos_vacios_eliminados": int(vacios.sum()),
        "claves_contradictorias": len(contradictorias),
        "filas_contradictorias_eliminadas": int(filas_contradictorias.sum()),
        "duplicados_eliminados": int(duplicados.sum()),
        "duplicados_por_etiqueta": datos.loc[duplicados, "label"].value_counts().to_dict(),
        "registros_limpios": len(limpio),
        "por_etiqueta_limpios": limpio["label"].value_counts().to_dict(),
    }
    return limpio, conteos


def particionar(limpio: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """60/20/20 estratificado con semilla fija. Como cada clave aparece una sola vez tras
    depurar, ninguna clave puede quedar en dos particiones; `verificar_sin_fuga` lo comprueba."""
    resto_frac = PROPORCIONES["validation"] + PROPORCIONES["test"]
    train, resto = train_test_split(
        limpio, test_size=resto_frac, stratify=limpio["label"], random_state=SEMILLA
    )
    validation, test = train_test_split(
        resto, test_size=PROPORCIONES["test"] / resto_frac, stratify=resto["label"], random_state=SEMILLA
    )
    return {"train": train, "validation": validation, "test": test}


def verificar_sin_fuga(particiones: dict[str, pd.DataFrame]) -> dict[str, int]:
    claves = {nombre: set(df["key"]) for nombre, df in particiones.items()}
    compartidas = {
        f"{a}&{b}": len(claves[a] & claves[b])
        for i, a in enumerate(PARTICIONES)
        for b in PARTICIONES[i + 1 :]
    }
    if any(compartidas.values()):
        raise ContratoInvalido(f"Mensajes equivalentes compartidos entre particiones: {compartidas}")
    return compartidas


def guardar(particiones: dict[str, pd.DataFrame], destino: Path = DIR_PROCESADOS) -> dict[str, Path]:
    destino.mkdir(parents=True, exist_ok=True)
    rutas = {}
    for nombre, df in particiones.items():
        rutas[nombre] = destino / f"{nombre}.parquet"
        df[["text", "label", "key"]].to_parquet(rutas[nombre], index=False)
    return rutas


def cargar_particiones(origen: Path = DIR_PROCESADOS) -> dict[str, pd.DataFrame]:
    """Lo que usan el entrenamiento y el notebook: nunca vuelven a depurar ni a partir."""
    faltantes = [p for p in PARTICIONES if not (origen / f"{p}.parquet").exists()]
    if faltantes:
        raise FileNotFoundError(f"Faltan particiones {faltantes}; corre `uv run python src/datos.py`")
    return {p: pd.read_parquet(origen / f"{p}.parquet") for p in PARTICIONES}


def preparar() -> dict:
    """Recorrido completo: descarga → carga → depuración → particiones → verificación → resumen."""
    ruta_corpus = descargar()
    crudo = cargar(ruta_corpus)
    limpio, conteos = depurar(crudo)
    particiones = particionar(limpio)
    compartidas = verificar_sin_fuga(particiones)
    guardar(particiones)

    meta_descarga = json.loads((DIR_CRUDOS / "descarga.json").read_text(encoding="utf-8"))
    resumen = {
        "fuente": FUENTE,
        "descarga": {**meta_descarga, "sha256_corpus": sha256(ruta_corpus)},
        "depuracion": conteos,
        "particiones": {
            "semilla": SEMILLA,
            "proporciones": PROPORCIONES,
            "tamanos": {n: len(df) for n, df in particiones.items()},
            "por_etiqueta": {n: df["label"].value_counts().to_dict() for n, df in particiones.items()},
            "proporcion_spam": {n: round(float(df["label"].eq("spam").mean()), 4) for n, df in particiones.items()},
            "claves_compartidas": compartidas,
        },
    }
    RUTA_RESUMEN.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESUMEN.write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resumen


if __name__ == "__main__":
    print(json.dumps(preparar(), indent=2, ensure_ascii=False))
