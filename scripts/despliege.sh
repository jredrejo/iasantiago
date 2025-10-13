#!/usr/bin/env bash
set -euo pipefail

BASE=/opt/iasantiago-rag

# TLS self-signed (ajusta si tienes tu propia CA)
mkdir -p $BASE/nginx/certs
if [ ! -f $BASE/nginx/certs/server.key ]; then
  openssl req -x509 -nodes -days 3650 -newkey rsa:4096 \
    -keyout $BASE/nginx/certs/server.key \
    -out $BASE/nginx/certs/server.crt \
    -subj "/CN=iasantiago.santiagoapostol.net/O=IASantiago/OU=IT"
  openssl dhparam -out $BASE/nginx/dhparam.pem 2048
fi

# Nginx conf
sudo ln -sf $BASE/nginx/nginx.conf /etc/nginx/nginx.conf
sudo systemctl enable --now nginx

# Python helper bench (opcional)
mkdir -p $BASE/scripts
cat > $BASE/scripts/bench_vllm.py <<'PY'
import time, asyncio, httpx, statistics
URL="http://127.0.0.1:8000/v1/chat/completions"
payload={"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Di hola en una frase."}]}
async def one():
    async with httpx.AsyncClient() as c:
        t0=time.time()
        r=await c.post(URL,json=payload,timeout=120)
        dt=time.time()-t0
        return dt
async def main(n=10):
    ts=[]
    for _ in range(n):
        ts.append(await one())
    print("N",n,"p50",statistics.median(ts),"avg",sum(ts)/n,"max",max(ts))
asyncio.run(main())
PY


# systemd units
sudo ln -sf $BASE/systemd/iasantiago-rag.service /etc/systemd/system/iasantiago-rag.service
sudo ln -sf $BASE/systemd/iasantiago-rag-eval.service /etc/systemd/system/iasantiago-rag-eval.service
sudo ln -sf $BASE/systemd/iasantiago-rag-eval.timer /etc/systemd/system/iasantiago-rag-eval.timer
sudo systemctl daemon-reload
sudo systemctl enable --now iasantiago-rag.service
sudo systemctl enable --now iasantiago-rag-eval.timer

# logrotate
sudo ln -sf $BASE/systemd/logrotate-telemetry /etc/logrotate.d/iasantiago-rag

echo "Despliegue completado. Abre https://iasantiago.santiagoapostol.net (acepta el certificado si es self-signed)."
