# Decisiones técnicas y uso de IA

Bitácora del examen final (Nimbo SMS). Cada decisión indica alternativas, razón y cómo se validó, con enlace al archivo o evidencia. El registro de IA conserva los prompts originales y resume sólo la parte relevante de cada respuesta.

## 1. Decisiones técnicas

### D1. Ambiente: un solo `uv.lock` con grupos por componente

- **Decisión:** proyecto `nimbo-sms` con Python 3.12 y cuatro grupos de dependencias (`train`, `api`, `ui`, `dev`) en [pyproject.toml](../pyproject.toml). Localmente se instalan todos (`default-groups = "all"`); cada imagen Docker instala sólo su grupo desde el mismo lock.
- **Alternativas:** un `pyproject.toml` por servicio (versiones que pueden contradecirse) o `requirements.txt` escritos a mano (el examen pide generarlos desde el lock).
- **Validación:** `uv sync --locked` sin cambios; las imágenes usan `uv sync --locked --only-group ...`.

### D2. `fastapi<0.142` por incompatibilidad con Prefect 3.8.8

- **Problema observado:** con FastAPI 0.162.4 el servidor de Prefect falla al iniciar con `TypeError: Body() got an unexpected keyword argument 'min_items'` (`prefect/server/api/concurrency_limits_v2.py`).
- **Decisión:** fijar `fastapi[standard]>=0.141.1,<0.142`, la combinación que ya funcionaba con Prefect 3.8.8 en la tarea 6.
- **Validación:** el servidor de Prefect arranca y el flujo completo termina en `Completed()`. Evidencia: captura del flujo en Prefect.
- **Límite:** la API usa la misma versión de FastAPI; si Prefect se actualiza, conviene revisar el límite.

### D3. Depuración y particiones

- **Decisión:** clave de equivalencia `texto.strip().lower()`, exclusión de claves contradictorias, primera aparición de duplicados y después partición 60/20/20 estratificada con semilla 42 ([src/datos.py](../src/datos.py)).
- **Por qué deduplicar antes de partir:** si un mensaje repetido cae en entrenamiento y en prueba, el modelo se evalúa con mensajes que ya vio. Al dejar cada clave una sola vez, ninguna puede repetirse entre particiones.
- **Resultado** ([datos-resumen.json](datos-resumen.json)): 5,574 leídos (4,827 ham / 747 spam); 0 vacíos; 0 claves contradictorias; 415 duplicados (310 ham / 105 spam); 5,159 limpios. Tamaños 3,095 / 1,032 / 1,032, con 12.4–12.5 % de spam en cada partición.
- **Validación:** `verificar_sin_fuga` interseca las claves de las tres particiones (resultado 0, 0, 0) y detiene el flujo con `ContratoInvalido` si alguna fuera distinta de cero.

### D4. Representación dentro del modelo registrado

- **Decisión:** `TfidfVectorizer` dentro de un `ColumnTransformer` sobre la columna `text`, en el mismo `Pipeline` que el clasificador ([src/entrenamiento.py](../src/entrenamiento.py)).
- **Por qué:** el vocabulario se aprende en cada `fit` (en tuning, sólo con entrenamiento) y viaja con el modelo. La API envía el mismo DataFrame `{"text": ...}` que se usó al evaluar, así que no hay un segundo preprocesamiento que pueda divergir.
- **Alternativa descartada:** pasar una lista de textos directamente al `Pipeline`. Funciona en Python, pero el esquema de MLflow describe un DataFrame con columna `text`, y un `TfidfVectorizer` que recibe un DataFrame recorre los nombres de columna en vez de los textos.

### D5. Métrica de selección, definida antes del tuning

- **Decisión:** F1 de `spam` como principal y precision de `spam` como desempate (constantes `METRICA_PRINCIPAL` y `METRICA_DESEMPATE` en [src/entrenamiento.py](../src/entrenamiento.py)).
- **Por qué:** con 12.4 % de spam, predecir siempre `ham` da accuracy de 0.875 y F1 de 0 (el baseline lo muestra en [metricas.csv](metricas.csv)). F1 equilibra spam no detectado (FN) y legítimos marcados (FP). El desempate favorece la precision porque la aplicación sólo sugiere: un FP hace desconfiar de un mensaje legítimo.

### D6. Champion elegido en validación aunque pierda en prueba

- **Resultado:** Naive Bayes ganó en validación (F1 0.9435 contra 0.9398 de la regresión logística) y quedó como `champion` (v1). En prueba la regresión logística obtuvo más (0.9639 contra 0.9474).
- **Decisión:** no cambiar el champion. La prueba se usa una sola vez para estimar desempeño; si se usara para elegir, dejaría de ser una estimación independiente (sección 7.5).
- **Lectura:** con 129 spam en validación, 0.004 de F1 equivale a uno o dos mensajes; los dos finalistas son prácticamente equivalentes y la diferencia en prueba también cabe en ese ruido.

### D7. `C` de la regresión logística en el borde del rango: se documenta, no se retunea

- **Observación:** el mejor `C` fue 96.8 con rango 0.1–100. Un óptimo en el borde sugiere que el rango era corto.
- **Decisión:** no ampliar el rango ni repetir el tuning, porque ya se habían visto los resultados de prueba y repetirlo sería seleccionar con información de prueba.
- **Mejora para otra iteración:** rango 0.1–1000, fijado antes de mirar prueba.

### D8. La API fija la versión al iniciar y valida con Pydantic estricto

- **Decisión:** al arrancar, [api/main.py](../api/main.py) resuelve `champion` a un número de versión y carga `models:/<modelo>/<versión>`, no `@champion`. Así lo que reporta `/health` es exactamente lo que se cargó, aunque el alias cambie mientras corre.
- **Contrato:** `extra="forbid"` rechaza campos adicionales; `strict=True` rechaza `{"text": 7}` en vez de convertirlo a `"7"`; `min_length=1`, `max_length=2000` y un validador para textos de sólo espacios. El texto se pasa al modelo sin modificar.
- **Si el modelo no carga:** la API arranca igual, pero `/health` y `/predict` responden 503, nunca 200 con un error.
- **Validación:** 5 casos inválidos → 422; 2,000 caracteres → 200 (límite incluido); paridad modelo/API en 3 textos inventados ([paridad.json](paridad.json), [src/paridad.py](../src/paridad.py)).

### D9. Contenedores: red bridge en vez de host

- **Decisión:** red por defecto de Compose; la UI llama a `http://api:8060` y los puertos se publican sólo en `127.0.0.1` ([compose.yaml](../compose.yaml)).
- **Alternativa:** `network_mode: host`, la referencia del curso y lo que usé en la clase 20. En Docker Desktop para Windows, `host` no publica los puertos en Windows como en Linux, y *bridge* funciona igual en ambos (ver [despliegue-aws.md](despliegue-aws.md)).
- **Puertos:** 8060 (API) y 8560 (UI), porque 8000 y 8501 ya los usan otros proyectos en mi máquina.
- **Sin secretos en imágenes:** `.dockerignore` excluye `.env`, `.git/`, `data/` y `artifacts/`; el `.env` llega sólo al contenedor de la API en tiempo de ejecución.

### D10. Imágenes sin caché de uv ni `chown -R`

- **Problema observado:** la primera construcción produjo imágenes de 2.1 GB (API) y 2.0 GB (UI) aunque sus ambientes ocupaban 518 MB y 461 MB. La caché de descargas de uv quedaba dentro de la imagen y `chown -R nimbo /app` copiaba el ambiente completo a una capa nueva.
- **Decisión:** crear el usuario antes de instalar (`COPY --chown`, `USER nimbo` antes de `uv sync`) y `UV_NO_CACHE=1`.
- **Validación:** 0.95 GB y 0.88 GB tras reconstruir; dentro de la imagen de la UI `import mlflow` falla, lo que confirma que sólo trae su grupo (34 paquetes contra 78 de la API).

### D11. Demostración en mi computadora de escritorio por Tailscale, no en AWS

- **Contexto:** Docker Desktop en mi laptop dejó de arrancar (se cayó WSL) el día anterior a la entrega.
- **Decisión:** construir y correr el mismo `compose.yaml` en mi computadora de escritorio (Docker Engine dentro de WSL, Linux), con puertos sólo en `127.0.0.1` y publicados únicamente en mi red privada de Tailscale (`tailscale serve`), sin abrir puertos del router. Se elimina al terminar el curso.
- **Alternativas:** EC2 (opcional en el examen, con costo y sin necesidad para sustentar) o reparar Docker Desktop contra reloj.
- **Validación:** desde la laptop, por la red privada: `/health` 200 con versión 1, los cinco casos 422, la UI responde 200 y `src/paridad.py` contra el contenedor da 3/3.

## 2. Uso de IA

### IA-1. Planeación del examen

```text
Herramienta y para qué la usé: Claude Code (Claude Opus 5.5) en VS Code, para leer el enunciado y proponer un orden de trabajo.
Prompt que utilicé: "tenemos que hacer el examen que está en docs/tareas/examen-final-2026, qué opinas?"
Qué propuso y qué acepté, modifiqué o rechacé, con mi razón: propuso reutilizar el patrón de las tareas 5 y 6 (Optuna anidado en MLflow, registro en Unity Catalog, tasks de Prefect con reintentos), F1 de spam como métrica y Databricks o MLflow local. Acepté Databricks porque ya lo tenía configurado y es el camino de referencia; acepté la métrica (ver D5).
Cómo lo comprobé, resultado y enlace a evidencia o archivo: comparé la propuesta con la rúbrica (sección 17) y con mis tareas 5 y 6. Evidencia: este documento y src/.
```

### IA-2. Ambiente, datos y flujo de entrenamiento

```text
Herramienta y para qué la usé: Claude Code, para escribir src/datos.py, src/entrenamiento.py y src/train_flow.py y la configuración del ambiente.
Prompt que utilicé: "Me parece bien tu flujo, usemos Databricks, creé la carpeta donde haremos todo del examen final, pero no repositorio" y después "listo, prosigamos".
Qué propuso y qué acepté, modifiqué o rechacé, con mi razón: propuso grupos de dependencias por servicio (D1), deduplicar antes de partir (D3), TF-IDF dentro de un ColumnTransformer (D4) y 20 trials TPE por familia. Acepté. Ante la falla de Prefect propuso fijar fastapi<0.142 (D2). Ante C en el borde, propuso no retunear (D7); lo acepté.
Cómo lo comprobé, resultado y enlace a evidencia o archivo: corrí `uv sync --locked` y `uv run python src/train_flow.py`; revisé en Databricks las dos versiones y sus aliases; los conteos de reports/datos-resumen.json coinciden con los 5,574 registros del enunciado. Evidencias: reports/evidencias/ (ver índice en el informe).
Comprobación propia: escribí paráfrasis de los 2 FP y 3 FN del champion y otros textos cortos inventados, y los probé en la UI desplegada. Noté que en textos cortos una sola palabra como "prize" o "reward" basta para que marque spam; hacen falta varias palabras típicas de ham (o desconocidas para el vocabulario) para compensarla. Es una limitación de la bolsa de palabras: esos términos pesan muchísimo en la decisión, y coincide con el patrón de los FP del informe (mensajes legítimos cortos). Evidencia: tabla de paráfrasis y análisis de errores en la sección 4 de examen-final.ipynb
```

### IA-3. API, interfaz, contenedores y explicación de AWS

```text
Herramienta y para qué la usé: Claude Code, para escribir api/main.py, src/paridad.py, ui/app.py, los Dockerfile, compose.yaml y reports/despliegue-aws.md.
Prompt que utilicé: "Está bien, documentemos y continuemos"
Qué propuso y qué acepté, modifiqué o rechacé, con mi razón: propuso cargar por número de versión (D8), Pydantic estricto, 503 si no hay modelo, red bridge en vez de host (D9) y t3.small para EC2. Acepté.
Cómo lo comprobé, resultado y enlace a evidencia o archivo: ejecuté con curl los cinco casos inválidos (422) y el válido (200), y src/paridad.py (3/3 coinciden). Evidencias: reports/evidencias/.
Hospedé todo en Pixilate (mi PC) y accedo por medio de tailscale, se puede pasar a cloudflare (con mi DNS personal) o a AWS de ser necesario en cualquier momento.
```

### IA-4. Despliegue para la sustentación e informe

```text
Herramienta y para qué la usé: Claude Code, para desplegar los contenedores en mi computadora de escritorio, corregir el tamaño de las imágenes y armar el informe (src/informe.py y las secciones de examen-final.ipynb) y la sección "Ejecución de mi solución" del README.
Prompt que utilicé: "a ver mira, quiero terminar esto asap, te parece? Se entrega en algunas horas pero quisiera que quedara corriendo en Pixilate para poder acceder a él desde tailscale para llegar, presentar, explicar y listo. Después de pasar la materia derribaríamos."
Qué propuso y qué acepté, modifiqué o rechacé, con mi razón: propuso copiar el árbol de trabajo en vez de clonar el repo privado en esa máquina (evita dejar credenciales de GitHub allá), restart: unless-stopped, D10 y D11. Para el análisis de errores propuso que las tablas del informe sólo muestren rasgos numéricos y que yo lea los mensajes con src/ver_errores.py y los parafrasee; lo acepté porque el enunciado prohíbe publicar mensajes y enviarlos a la IA.
Cómo lo comprobé, resultado y enlace a evidencia o archivo: el notebook se ejecutó completo con kernel nuevo (0 errores); contrato y paridad contra el contenedor (D11). Revisé y escribí las paráfrasis de la sección 4 leyendo los mensajes localmente.
```

## 3. Límites reconocidos

- El corpus es histórico, sólo en inglés y con 642 spam tras depurar: no demuestra eficacia con mensajes actuales, otros idiomas ni todos los tipos de fraude.
- La comparación entre finalistas se apoya en una sola partición de validación; una validación cruzada daría una idea de la varianza.
- D7: el rango de `C` quedó corto.
