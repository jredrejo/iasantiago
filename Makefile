SHELL := /bin/bash
ENV_FILE := .env
BACKUP_DIR := /opt/iasantiago-rag/backups
DATE := $(shell date +%F_%H%M%S)

.PHONY: gen-env up down stop seed reset bench \
        status logs tail rag-restart \
        backup backup-topics backup-qdrant backup-whoosh restore \
        watcher-on watcher-off \
        publish-docs unpublish-docs \
        eval-sample eval-file eval-nightly \
        ingest web


gen-env:
	@if [ ! -f $(ENV_FILE) ]; then \
	    echo "Creando .env desde .env.example"; \
	    cp .env.example .env; \
	fi

up: gen-env
	docker compose pull
	docker compose build
	docker compose up -d

down:
	docker compose down
stop:
	docker compose stop

seed: gen-env
	@echo "Creando carpetas de temas y copiando PDFs de ejemplo..."
	@mkdir -p data/storage data/whoosh
	@mkdir -p topics/Chemistry topics/Electronics topics/Programming
	@cp -r seeds/* topics/
	@echo "Lanzando ingestor para indexar..."
	docker compose restart ingestor

reset: gen-env
	@echo "Reseteando Qdrant y Whoosh..."
	@docker compose down
	@sudo rm -rf data/whoosh/.processing_state.json
	@sudo rm -rf data/storage/*
	@sudo rm -rf data/whoosh/*
	@docker compose up -d ingestor

bench:
	@echo "Benchmark de vLLM /chat/completions"
	python3 scripts/bench_vllm.py || true

# ------- Utilidad -------
status:
	docker compose ps

logs:
	docker compose logs --tail=200

tail:
	docker compose logs -f

rag-restart:
	docker compose restart rag-api

# ------- Modo ingestión / web -------
ingest:
	@echo "Deteniendo vllm, rag-api, openwebui y lanzando ingestor..."
	docker compose stop vllm rag-api openwebui oauth2-proxy || true
	docker compose up -d ingestor

web:
	@echo "Deteniendo ingestor y lanzando oauth2-proxy..."
	docker compose stop ingestor || true
	docker compose up -d oauth2-proxy

# ------- Backups -------
backup: backup-topics backup-qdrant backup-whoosh
	@echo "Backup completo en $(BACKUP_DIR)/full-$(DATE)"

backup-topics:
	@sudo mkdir -p $(BACKUP_DIR)
	sudo tar -C /opt/iasantiago-rag -czf $(BACKUP_DIR)/topics-$(DATE).tgz topics
	@echo "[OK] topics -> $(BACKUP_DIR)/topics-$(DATE).tgz"

backup-qdrant:
	@sudo mkdir -p $(BACKUP_DIR)
	sudo tar -C /opt/iasantiago-rag -czf $(BACKUP_DIR)/qdrant-$(DATE).tgz data/storage
	@echo "[OK] qdrant -> $(BACKUP_DIR)/qdrant-$(DATE).tgz"

backup-whoosh:
	@sudo mkdir -p $(BACKUP_DIR)
	sudo tar -C /opt/iasantiago-rag -czf $(BACKUP_DIR)/whoosh-$(DATE).tgz data/whoosh
	@echo "[OK] whoosh -> $(BACKUP_DIR)/whoosh-$(DATE).tgz"

# Uso:
# make restore BACKUP=2025-10-10_023000
restore:
	@if [ -z "$(BACKUP)" ]; then echo "ERROR: usa make restore BACKUP=YYYY-MM-DD_HHMMSS"; exit 1; fi
	@echo "Restaurando backup $(BACKUP) (se requiere ventana de mantenimiento)..."
	docker compose down
	sudo rm -rf data/storage/* data/whoosh/*
	sudo tar -C /opt/iasantiago-rag -xzf $(BACKUP_DIR)/qdrant-$(BACKUP).tgz
	sudo tar -C /opt/iasantiago-rag -xzf $(BACKUP_DIR)/whoosh-$(BACKUP).tgz
	@echo "Opcionalmente restaura topics si procede:"
	@echo "  sudo tar -C /opt/iasantiago-rag -xzf $(BACKUP_DIR)/topics-$(BACKUP).tgz"
	docker compose up -d

# ------- Watcher on/off (override de comando en ingestor) -------
# Usa un archivo de override que cambia el command a watcher.py
watcher-on:
	@echo "Activando watcher (indexación en caliente)..."
	@if [ ! -f docker-compose.watcher.yml ]; then \
		bash -c 'cat > docker-compose.watcher.yml <<YML\n\
version: "3.9"\n\
services:\n\
  ingestor:\n\
    command: ["python", "-u", "watcher.py"]\n\
YML'; \
	fi
	COMPOSE_FILE=docker-compose.yml:docker-compose.watcher.yml docker compose up -d --build ingestor
	@echo "[OK] watcher activo"

watcher-off:
	@echo "Desactivando watcher (vuelve a escaneo inicial)..."
	COMPOSE_FILE=docker-compose.yml:docker-compose.watcher.yml docker compose stop ingestor || true
	docker compose up -d --build ingestor
	@echo "[OK] watcher desactivado (ingestor = main.py)"


# ------- Publicar PDFs como /docs bajo Nginx -------
publish-docs:
	@echo "Publicando /docs/ en Nginx..."
	@sudo bash -c 'tee /etc/nginx/conf.d/iasantiago-docs.conf >/dev/null <<NGINX\n\
location /docs/ {\n\
    autoindex on;\n\
    alias /opt/iasantiago-rag/topics/;\n\
}\n\
NGINX'
	sudo nginx -t
	sudo systemctl reload nginx
	@echo "[OK] /docs/ disponible en https://iasantiago.santiagoapostol.net/docs/"

unpublish-docs:
	@echo "Despublicando /docs/..."
	@sudo rm -f /etc/nginx/conf.d/iasantiago-docs.conf
	sudo nginx -t
	sudo systemctl reload nginx
	@echo "[OK] /docs/ retirado"

# ------- Evaluación offline -------
# Plantilla grande de casos -> eval/cases.sample.json (ver más abajo)
eval-sample:
	@echo "Ejecutando evaluación con plantilla..."
	curl -s http://127.0.0.1:8001/v1/eval/offline \
	    -H 'Content-Type: application/json' \
	    --data-binary @eval/cases.sample.json | jq . > eval/last_eval.json
	@echo "[OK] Resultado en eval/last_eval.json"

# Uso:
# make eval-file FILE=eval/mis_casos.json
eval-file:
	@if [ -z "$(FILE)" ]; then echo "ERROR: usa make eval-file FILE=path.json"; exit 1; fi
	curl -s http://127.0.0.1:8001/v1/eval/offline \
	    -H 'Content-Type: application/json' \
	    --data-binary @$(FILE) | jq . > eval/last_eval.json
	@echo "[OK] Resultado en eval/last_eval.json"

# Lanza el service systemd de evaluación nocturna ahora mismo
eval-nightly:
	sudo systemctl start iasantiago-rag-eval.service && sudo journalctl -u iasantiago-rag-eval.service -n 50 --no-pager
