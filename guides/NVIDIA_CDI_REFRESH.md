# Arreglar el error de montaje de librerías NVIDIA en Docker (spec CDI obsoleto)

Guía para resolver el fallo al arrancar contenedores con GPU después de una
actualización del driver NVIDIA en el host.

## 1. Síntoma

Al levantar un contenedor con GPU, Docker falla nada más crearlo:

```
$ docker compose up -d vllm
Error response from daemon: failed to create task for container: failed to create shim task:
OCI runtime create failed: runc create failed: unable to start container process:
error during container init: failed to fulfil mount request:
open /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.595.71.05: no such file or directory
```

La librería puede ser cualquier otra (`libnvidia-ml.so.X`, `libcuda.so.X`…), pero el
patrón es siempre el mismo: **runc intenta montar un fichero `.so` con un número de
versión de driver que ya no existe en el host**.

## 2. Causa

El toolkit de NVIDIA describe la GPU a Docker mediante un fichero **CDI**
(*Container Device Interface*) en `/var/run/cdi/nvidia.yaml`. Ese fichero contiene
**rutas absolutas con la versión del driver incrustada**, por ejemplo:

```yaml
- hostPath: /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.595.71.05
```

Cuando `apt` actualiza el driver, las librerías viejas se borran y aparecen las nuevas
(`...so.595.84`), pero **el spec CDI no se regenera solo si el servicio encargado de
hacerlo ha fallado**. Resultado: el spec apunta a ficheros inexistentes y runc revienta.

El servicio responsable es `nvidia-cdi-refresh.service`, disparado por
`nvidia-cdi-refresh.path` cuando cambian `/lib/modules/$(uname -r)/modules.dep*` o
`/usr/bin/nvidia-ctk`. Es frágil: si el `ExecStart` falla durante la actualización del
paquete (típicamente `status=127`, binario no encontrado porque `apt` está a medias),
systemd reintenta 5 veces en 10 segundos, agota el `StartLimitBurst` y **deja tanto el
`.service` como el `.path` en estado `failed`**. A partir de ese momento nadie vigila y
el spec se queda congelado indefinidamente.

> Ocurrió así el 2026-07-25 en esta máquina: driver actualizado a 595.84, spec CDI
> congelado en 595.71.05 desde el arranque del 2026-06-26.

## 3. Cuándo hay que aplicar esta guía

Aplícala cuando se cumpla **cualquiera** de estas condiciones:

- Un contenedor con GPU falla con `failed to fulfil mount request: open …libXXX_nvidia.so.<versión>`.
- Acabas de hacer `apt upgrade` / `unattended-upgrade` que ha tocado paquetes `nvidia-*`.
- `systemctl is-failed nvidia-cdi-refresh.service` devuelve `failed`.

**No** hace falta si el error es otro (OOM de GPU, puerto ocupado, modelo que no cabe):
esta guía sólo cubre el desajuste de versiones entre el spec CDI y el driver instalado.

### Contenedores afectados en este proyecto

Los que reservan GPU en `docker-compose.yml`: **`vllm`**, **`rag-api`** e **`ingestor`**.
El `llama-server` de llama.cpp corre *baremetal* (no en contenedor), así que **no** se ve
afectado por esto.

## 4. Diagnóstico (30 segundos)

Compara las tres versiones que deberían coincidir:

```bash
# 1. Versión del driver realmente instalada y cargada
nvidia-smi --query-gpu=driver_version --format=csv,noheader
cat /proc/driver/nvidia/version

# 2. Versión de las librerías presentes en disco
ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*

# 3. Versión a la que apunta el spec CDI
grep -o 'libnvidia-ml\.so\.[0-9.]*' /var/run/cdi/nvidia.yaml | sort -u
```

Si (3) no coincide con (1) y (2), es exactamente este problema.

Comprueba además por qué no se refrescó solo:

```bash
systemctl status nvidia-cdi-refresh.path nvidia-cdi-refresh.service --no-pager -l
```

Un `Active: failed (Result: unit-start-limit-hit)` confirma el diagnóstico.

## 5. Solución paso a paso

### Paso 1 — Regenerar el spec CDI

```bash
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
```

Los avisos `Could not locate nvidia-imex` / `nvidia-imex-ctl` son **normales** (son
herramientas de multinodo que no usamos). Lo que importa es la línea final
`Generated CDI spec with version 0.7.0`.

Verifica que ahora la versión es la correcta (debe coincidir con `nvidia-smi`):

```bash
grep -o 'libnvidia-ml\.so\.[0-9.]*' /var/run/cdi/nvidia.yaml | sort -u
```

> **Ojo con la ruta de salida.** El fichero debe escribirse en `/var/run/cdi/nvidia.yaml`,
> que es la ruta que usa el propio `nvidia-cdi-refresh.service` y que está en la lista
> `spec-dirs` de `/etc/nvidia-container-runtime/config.toml`. No lo generes también en
> `/etc/cdi/`: tendrías dos specs declarando el mismo dispositivo `nvidia.com/gpu` y el
> runtime se quejaría de nombres duplicados.

### Paso 2 — Recrear los contenedores con GPU

Levanta los contenedores usando `--force-recreate`:

```bash
cd /opt/iasantiago-rag
docker compose up -d --force-recreate vllm
```

Si hay más contenedores con GPU parados por lo mismo:

```bash
docker compose up -d --force-recreate vllm rag-api
docker compose --profile ingest up -d --force-recreate ingestor   # sólo si toca ingestar
```

> **¿Por qué `--force-recreate` y no un `restart`?** Es la vía que se ha verificado que
> funciona: en el incidente del 2026-07-27 un `docker compose up -d vllm` sobre el
> contenedor ya existente falló, y recrearlo lo resolvió. Es posible que tras regenerar
> el spec baste con un arranque normal (el daemon resuelve los dispositivos GPU al
> arrancar la tarea), pero recrear el contenedor es barato, no destruye datos —los
> volúmenes se conservan— y elimina la duda. Ante este error, recrea y no investigues.

### Paso 3 — Reactivar el refresco automático

Mientras las unidades sigan en `failed`, la próxima actualización de driver volverá a
dejarte el spec obsoleto sin avisar:

```bash
sudo systemctl reset-failed nvidia-cdi-refresh.service nvidia-cdi-refresh.path
sudo systemctl start nvidia-cdi-refresh.path
systemctl is-active nvidia-cdi-refresh.path   # debe decir: active
```

### Paso 4 — Verificar

```bash
# El contenedor arranca y pasa el healthcheck (vLLM tarda ~2 min en cargar el modelo)
docker ps --filter name=vllm --format '{{.Status}}'

# La GPU se ve desde dentro del contenedor
docker exec vllm nvidia-smi

# El endpoint responde
curl -s http://localhost:8000/v1/models | head
```

## 6. Resumen ejecutable

Si ya sabes que es esto, son cuatro órdenes:

```bash
sudo nvidia-ctk cdi generate --output=/var/run/cdi/nvidia.yaml
sudo systemctl reset-failed nvidia-cdi-refresh.service nvidia-cdi-refresh.path
sudo systemctl start nvidia-cdi-refresh.path
docker compose -f /opt/iasantiago-rag/docker-compose.yml up -d --force-recreate vllm
```

## 7. Prevención

- **Después de cada `apt upgrade` que toque paquetes `nvidia-*`**, comprueba
  `systemctl is-failed nvidia-cdi-refresh.service` antes de dar por buena la
  actualización.
- Si actualizas el driver, lo más limpio es **reiniciar la máquina**: al arrancar se
  regenera el spec y todos los contenedores se crean de nuevo con las rutas correctas.
  Sin reinicio, hay que hacer el paso 1 y el paso 2 a mano.
- No edites `/var/run/cdi/nvidia.yaml` a mano para "arreglar" los números de versión:
  cambia también la lista de ficheros según la versión del driver. Regenéralo siempre
  con `nvidia-ctk`.

## 8. Referencias

- Unidades systemd: `/etc/systemd/system/nvidia-cdi-refresh.{service,path}`
- Configuración del runtime: `/etc/nvidia-container-runtime/config.toml`
  (clave `spec-dirs`, que decide dónde se buscan los specs)
- Spec CDI activo: `/var/run/cdi/nvidia.yaml`
