# TLS Configuration with Let's Encrypt
  server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name $ALL_DOMAINS;#!/bin/bash

# Script de configuración HTTPS multi-dominio con Let's Encrypt y nginx
# Adaptado para configuración existente de Open WebUI + RAG API
# Uso: sudo bash script.sh

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Variables de configuración
# Solo dominios públicos para Let's Encrypt
DOMAINS=("ia.iessaenzdeburuaga.es" )
MAIN_DOMAIN="ia.iessaenzdeburuaga.es"
# Todos los nombres (incluyendo local) para nginx
ALL_DOMAINS="ia.iessaenzdeburuaga.es"
CERT_PATH="/opt/iasantiago-rag/nginx/certs"
LETSENCRYPT_PATH="/etc/letsencrypt/live/$MAIN_DOMAIN"
NGINX_CONF="/etc/nginx/nginx.conf"

echo -e "${GREEN}=== Configuración HTTPS Multi-dominio para Open WebUI ===${NC}\n"

# 1. Verificar si es root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Este script debe ejecutarse como root (sudo)${NC}"
   exit 1
fi

# 2. Hacer backup de configuración actual
echo -e "${YELLOW}[1/8] Haciendo backup de configuración actual...${NC}"
cp $NGINX_CONF "${NGINX_CONF}.backup.$(date +%Y%m%d-%H%M%S)"
cp /etc/nginx/sites-available/* /etc/nginx/sites-available.backup/ 2>/dev/null || true

# 3. Actualizar sistema
echo -e "${YELLOW}[2/8] Actualizando sistema...${NC}"
apt-get update
apt-get upgrade -y

# 4. Instalar dependencias
echo -e "${YELLOW}[3/8] Instalando dependencias...${NC}"
apt-get install -y nginx certbot python3-certbot-nginx openssl

# 5. Crear estructura de directorios
echo -e "${YELLOW}[4/8] Creando estructura de directorios...${NC}"
mkdir -p /etc/nginx/sites-available
mkdir -p /etc/nginx/sites-enabled
mkdir -p /var/www/certbot
mkdir -p /etc/nginx/snippets

# 6. Crear snippets reutilizables
echo -e "${YELLOW}[5/8] Creando snippets de configuración...${NC}"

cat > /etc/nginx/snippets/ssl-params.conf << 'EOF'
# SSL Parameters
ssl_session_timeout 1d;
ssl_session_cache shared:MozSSL:10m;
ssl_session_tickets off;

ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
ssl_prefer_server_ciphers off;

# OCSP stapling
ssl_stapling on;
ssl_stapling_verify on;

add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "no-referrer-when-downgrade" always;
EOF

cat > /etc/nginx/snippets/streaming-proxy.conf << 'EOF'
# WebSocket support
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";

# Disable buffering for streaming/SSE
proxy_buffering off;
proxy_cache off;
proxy_request_buffering off;
proxy_set_header X-Accel-Buffering no;

# Headers
proxy_set_header Host $host;
proxy_set_header X-Real-IP $remote_addr;
proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;

# Timeouts
proxy_read_timeout 300s;
proxy_connect_timeout 75s;
proxy_send_timeout 300s;

# Chunked encoding
chunked_transfer_encoding on;
EOF

# 7. Solicitar certificado con Let's Encrypt (DNS challenge)
echo -e "${YELLOW}[6/8] Solicitando certificado a Let's Encrypt...${NC}"
echo -e "${YELLOW}Se abrirá el asistente interactivo para validación DNS${NC}\n"

# Construir argumentos de dominio
DOMAIN_ARGS=""
for domain in "${DOMAINS[@]}"; do
    DOMAIN_ARGS="$DOMAIN_ARGS -d $domain"
done

certbot certonly --manual --preferred-challenges=dns $DOMAIN_ARGS

# 8. Crear nuevo nginx.conf con Let's Encrypt
echo -e "${YELLOW}[7/8] Actualizando configuración de nginx...${NC}"

cat > $NGINX_CONF << EOF
user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
  worker_connections 1024;
}

http {
  include       /etc/nginx/mime.types;
  default_type  application/octet-stream;

  sendfile on;
  tcp_nopush on;
  tcp_nodelay on;
  keepalive_timeout 65;
  types_hash_max_size 2048;

  # Global settings for streaming/SSE
  proxy_buffering off;
  proxy_request_buffering off;
  proxy_http_version 1.1;

  # Buffer sizes (keep small for streaming)
  proxy_buffer_size 4k;
  proxy_buffers 8 4k;
  proxy_busy_buffers_size 8k;

  # Upstream definitions
  upstream openwebui {
    server 127.0.0.1:8080;
    keepalive 32;
  }

  upstream rag-api {
    server 127.0.0.1:8001;
    keepalive 32;
  }

  # TLS Configuration with Let's Encrypt
  server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name ia.iessaenzdeburuaga.es  ia.iessaenzdeburuaga;

    access_log /var/log/nginx/openwebui_access.log;
    error_log /var/log/nginx/openwebui_error.log;

    # Let's Encrypt certificates
    ssl_certificate     $LETSENCRYPT_PATH/fullchain.pem;
    ssl_certificate_key $LETSENCRYPT_PATH/privkey.pem;

    # Include SSL parameters snippet
    include /etc/nginx/snippets/ssl-params.conf;

    # Callback OAuth para OpenWebUI
    location = /_auth/callback {
      proxy_pass http://openwebui/_auth/callback;
      include /etc/nginx/snippets/streaming-proxy.conf;
    }

    # Main location - Open WebUI
    location / {
      proxy_pass http://openwebui/;
      include /etc/nginx/snippets/streaming-proxy.conf;
    }

    # Documentos estáticos
    location /docs/ {
      autoindex on;
      alias /opt/iasantiago-rag/topics/;

      # Security headers
      add_header X-Content-Type-Options nosniff;
      add_header X-Frame-Options DENY;
    }
  }

  # HTTP to HTTPS redirect
  server {
    listen 80;
    listen [::]:80;
    server_name ia.iessaenzdeburuaga.es ia.iessaenzdeburuaga;

    # Allow Let's Encrypt ACME challenge
    location /.well-known/acme-challenge/ {
      root /var/www/certbot;
    }

    # Redirect everything else to HTTPS
    location / {
      return 301 https://\$server_name\$request_uri;
    }
  }
}
EOF

# Validar configuración nginx
echo -e "${YELLOW}Validando configuración de nginx...${NC}"
nginx -t

# Recargar nginx
systemctl reload nginx

# 9. Configurar renovación automática
echo -e "${YELLOW}[8/8] Configurando renovación automática de certificados...${NC}"

systemctl enable certbot.timer
systemctl start certbot.timer

# Crear hook para recargar nginx después de renovación
mkdir -p /etc/letsencrypt/renewal-hooks/post
cat > /etc/letsencrypt/renewal-hooks/post/nginx.sh << 'EOF'
#!/bin/bash
# Copiar certificados a ruta personalizada
cp /etc/letsencrypt/live/ia.iessaenzdeburuaga.es/fullchain.pem /opt/iasantiago-rag/nginx/certs/server.crt
cp /etc/letsencrypt/live/ia.iessaenzdeburuaga.es/privkey.pem /opt/iasantiago-rag/nginx/certs/server.key
chmod 644 /opt/iasantiago-rag/nginx/certs/server.crt
chmod 600 /opt/iasantiago-rag/nginx/certs/server.key
# Recargar nginx
systemctl reload nginx
EOF
chmod +x /etc/letsencrypt/renewal-hooks/post/nginx.sh

# Copiar certificados iniciales
mkdir -p $CERT_PATH
cp $LETSENCRYPT_PATH/fullchain.pem $CERT_PATH/server.crt
cp $LETSENCRYPT_PATH/privkey.pem $CERT_PATH/server.key
chmod 644 $CERT_PATH/server.crt
chmod 600 $CERT_PATH/server.key

# Mostrar resumen
echo -e "\n${GREEN}=== Configuración completada ===${NC}\n"
echo -e "${GREEN}✓ nginx actualizado y configurado${NC}"
echo -e "${GREEN}✓ Certificados Let's Encrypt en $CERT_PATH${NC}"
echo -e "${GREEN}✓ Renovación automática habilitada${NC}"
echo -e "${GREEN}✓ Snippets de configuración creados${NC}\n"

echo -e "${YELLOW}Información de los certificados:${NC}"
certbot certificates

echo -e "\n${YELLOW}Dominios configurados:${NC}"
for domain in "${DOMAINS[@]}"; do
    echo "  ✓ https://$domain (certificado Let's Encrypt)"
done
echo "  ✓ https://ia.santiago (usa mismo certificado vía SNI)"

echo -e "\n${YELLOW}Próximos pasos:${NC}"
echo "1. Verifica que Open WebUI corre en puerto 8080"
echo "2. Verifica que RAG API corre en puerto 8001"
echo "3. Prueba el acceso: https://ia.iessaenzdeburuaga.es"
echo "4. Prueba desde intranet: https://ia.santiago"
echo ""
echo -e "${YELLOW}Para verificar logs:${NC}"
echo "  - tail -f /var/log/nginx/openwebui_access.log"
echo "  - tail -f /var/log/nginx/openwebui_error.log"
echo ""
echo -e "${YELLOW}Para probar renovación:${NC}"
echo "  - sudo certbot renew --dry-run"
echo ""
echo -e "${YELLOW}Backup de configuración anterior:${NC}"
echo "  - $NGINX_CONF.backup*"
echo ""
echo -e "${GREEN}¡Listo! HTTPS está activo en tus tres dominios${NC}"
