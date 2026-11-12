"""Interfaz de Nimbo SMS: consulta la API por HTTP. No carga el modelo ni credenciales.

Uso local: `uv run streamlit run ui/app.py --server.port 8560 --server.address 127.0.0.1`
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8060").rstrip("/")
MAX_CARACTERES = 2000
TIMEOUT = 10

st.set_page_config(page_title="Nimbo SMS", page_icon="✉️")
st.title("Nimbo SMS")
st.write(
    "Pega un SMS **en inglés** para obtener una **estimación** de si es `ham` (legítimo) o "
    "`spam` (no solicitado). El resultado apoya tu decisión: no bloquea ni elimina mensajes."
)
st.caption("Modelo entrenado con un corpus histórico (2011) en inglés; puede fallar con mensajes actuales u otros idiomas.")

texto = st.text_area("Mensaje SMS", max_chars=MAX_CARACTERES, height=150,
                     placeholder="Can we meet after class?")

if st.button("Clasificar", type="primary"):
    if not texto.strip():
        st.warning("Escribe un mensaje: no puede estar vacío ni contener sólo espacios.")
    else:
        try:
            respuesta = requests.post(f"{API_URL}/predict", json={"text": texto}, timeout=TIMEOUT)
        except requests.RequestException:
            st.error(f"No se pudo conectar con la API en {API_URL}. Verifica que esté en ejecución.")
        else:
            if respuesta.status_code == 200:
                datos = respuesta.json()
                if datos["label"] == "spam":
                    st.error("Estimación: **spam** (posible mensaje no solicitado)")
                else:
                    st.success("Estimación: **ham** (mensaje legítimo)")
                st.caption(f"Modelo `{datos['model_name']}` · versión **{datos['model_version']}**")
            elif respuesta.status_code == 422:
                st.warning("La API rechazó el texto (HTTP 422): debe tener entre 1 y 2,000 caracteres y no sólo espacios.")
            elif respuesta.status_code == 503:
                st.error("La API está activa, pero el modelo no está cargado (HTTP 503).")
            else:
                st.error(f"Respuesta inesperada de la API: HTTP {respuesta.status_code}.")
