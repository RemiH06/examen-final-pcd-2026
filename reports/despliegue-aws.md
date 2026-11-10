# Despliegue en AWS EC2 (explicación aplicada)

Cómo se trasladaría Nimbo SMS a la instancia EC2 del laboratorio (clase 20) en **`us-east-1`**. El despliegue real es opcional y no se realizó; este documento describe el recorrido, la verificación, los costos y el cierre. No incluye IP, llaves, datos de cuenta ni credenciales.

## 1. Instancia

| Aspecto | Elección | Razón |
|---|---|---|
| Región | `us-east-1` | Indicada por el examen; es donde está la instancia del laboratorio. |
| Sistema | Ubuntu Server LTS (64 bits, x86_64) | Docker Engine y el plugin de Compose tienen paquetes oficiales; las imágenes `python:3.12-slim` son amd64. |
| Capacidad | `t3.small` (2 vCPU, 2 GiB) | La API carga scikit-learn, pandas y el cliente de MLflow; construir las dos imágenes con 1 GiB (`t3.micro`) puede quedarse sin memoria. El modelo TF-IDF + Naive Bayes ocupa pocos MB en memoria. |
| Disco | EBS `gp3` de 20 GiB | Sistema (~4 GiB) + capas de Docker de ambas imágenes (~1.5–2 GiB) + caché de build, con margen. |

> Si la instancia del laboratorio es de otro tipo, se reutiliza tal cual (sección 12 del examen) y sólo se vigila la memoria durante `docker compose build` con `free -h`.

## 2. Acceso por SSH

- Grupo de seguridad: regla **TCP 22** con origen **My IP** (`x.x.x.x/32`), nunca `0.0.0.0/0`. Si cambió la IP de mi computadora, se edita el origen de esa regla antes de conectar.
- Se consulta la **dirección pública actual** de la instancia en la consola (cambia cada vez que se detiene y arranca, salvo con Elastic IP).
- Conexión con la llave del laboratorio, que vive fuera de cualquier repositorio y con permisos restrictivos:

```bash
chmod 400 ~/.ssh/llave-laboratorio.pem
ssh -i ~/.ssh/llave-laboratorio.pem ubuntu@<IP_PUBLICA_EC2>
```

## 3. Código: clon del repositorio privado

El repositorio es privado, así que el servidor necesita una credencial de **sólo lectura** y sólo para este repo: una *deploy key* SSH generada en la instancia (o un token *fine-grained* con permiso de lectura de contenido). No se copian credenciales personales al servidor.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/nimbo_deploy -N ""
cat ~/.ssh/nimbo_deploy.pub   # se agrega en GitHub → Settings → Deploy keys (read-only)
GIT_SSH_COMMAND="ssh -i ~/.ssh/nimbo_deploy" git clone git@github.com:RemiH06/examen-final-pcd-2026.git
cd examen-final-pcd-2026 && git switch main && git pull
```

Se despliega `main` (la versión fusionada en el PR).

## 4. Docker y Compose

```bash
sudo apt-get update
# Instalación oficial de Docker Engine + compose plugin (docs.docker.com/engine/install/ubuntu)
sudo docker version && sudo docker compose version
```

Las imágenes se construyen en el servidor con los mismos `api/Dockerfile` y `ui/Dockerfile`; las dependencias salen del mismo `uv.lock`, por eso el ambiente remoto coincide con el local.

## 5. Configuración externa

El `.env` no está en Git ni en las imágenes. Se crea en el servidor desde la plantilla:

```bash
cp .env.example .env
chmod 600 .env
nano .env   # DATABRICKS_HOST, DATABRICKS_TOKEN, MODEL_NAME, MODEL_ALIAS
```

Sólo el servicio `api` recibe el `.env` (`env_file` en [compose.yaml](../compose.yaml)). La UI recibe únicamente `API_URL`.

## 6. Red: localhost frente a dirección remota

- **Entre contenedores:** en la red *bridge* de Compose la UI llama a la API por nombre de servicio, `http://api:8060`. Esto funciona igual en Windows y en Linux, por eso no cambia al pasar a EC2.
- **Dentro del servidor:** la API se publica en `127.0.0.1:8060` del host. Sirve para comprobar con `curl` desde la sesión SSH, pero no es accesible desde Internet.
- **Desde mi navegador:** la UI se publica en el host. En EC2 su puerto se cambia a `0.0.0.0:8560:8560` (un archivo `compose.override.yaml` sólo en el servidor) y el grupo de seguridad decide quién entra.
- **Alternativa de la clase 20:** `network_mode: host`, con la API escuchando en `127.0.0.1` y `API_URL=http://127.0.0.1:8060`. En Linux es equivalente; localmente se eligió *bridge* porque en Docker Desktop para Windows `host` no expone los puertos a Windows igual que en Linux.

## 7. Reglas de acceso (grupo de seguridad)

| Puerto | Protocolo | Origen | Para qué |
|---|---|---|---|
| 22 | TCP | My IP | SSH |
| 8560 | TCP | My IP | Interfaz Streamlit |
| 8060 | — | sin regla | La API no se expone; la UI la alcanza por la red interna de Docker |

## 8. Arranque y verificación remota

```bash
sudo docker compose up --build -d
sudo docker compose ps                              # api "healthy", ui "running"
curl -s http://127.0.0.1:8060/health                # 200 con model_version
curl -s -X POST http://127.0.0.1:8060/predict -H "Content-Type: application/json" \
     -d '{"text": "Can we meet after class?"}'     # 200 con label y model_version
sudo docker compose logs api --tail 50              # "Modelo ...@champion → versión N cargado"
```

Después, desde mi computadora: `http://<IP_PUBLICA_EC2>:8560`, con un texto inventado, comprobando que la versión mostrada coincide con `/health`.

## 9. Costos (us-east-1, bajo demanda, referencia de precios públicos)

| Recurso | Costo aproximado | ¿Cobra con la instancia detenida? |
|---|---|---|
| `t3.small` | ~0.021 USD/hora (~0.50 USD/día) | No |
| EBS gp3 20 GiB | ~0.08 USD/GiB-mes → ~1.60 USD/mes | **Sí**, mientras el volumen exista |
| IPv4 pública | ~0.005 USD/hora mientras esté asignada | Una Elastic IP sin asociar o asociada a una instancia detenida **sí** cobra |
| Snapshots de EBS / AMIs | ~0.05 USD/GiB-mes | **Sí** |
| Transferencia de salida | Primeros 100 GB/mes sin costo | — |

Una sesión de prueba de 2 horas cuesta del orden de 0.05 USD de cómputo. El costo que se olvida es el que sigue después: disco, IP elástica y snapshots.

## 10. Cierre

```bash
sudo docker compose down          # detiene y elimina contenedores y la red de Compose
sudo docker image prune -f        # opcional: libera disco
exit
```

En la consola de EC2:

- **Detener (Stop):** se apaga la instancia y deja de cobrar cómputo; el volumen EBS se conserva y **sigue cobrando**. La IP pública cambia al volver a arrancar. Es la opción si se va a reutilizar.
- **Terminar (Terminate):** elimina la instancia y, si *Delete on termination* está activo, también su volumen raíz. Es irreversible. Hay que revisar después que no queden volúmenes sueltos, Elastic IPs, snapshots ni grupos de seguridad sin uso.

Verificación final: estado `stopped` (o `terminated`) en la consola, sección **Volumes** sin volúmenes `available` olvidados y **Elastic IPs** vacía. No se mantiene la instancia encendida para la sustentación.
