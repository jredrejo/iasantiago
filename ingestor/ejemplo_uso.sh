#!/bin/bash
# Ejemplos de uso del sistema de caché SQLite para LLaVA

set -e

PROJECT_NAME="tu_proyecto"  # Cambiar según tu proyecto
INGESTOR_CONTAINER="${PROJECT_NAME}-ingestor"
CACHE_VOLUME="${PROJECT_NAME}_llava_cache"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  EJEMPLOS DE USO - SISTEMA DE CACHÉ SQLite PARA LLaVA         ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# ============================================================
# 1. VER ESTADÍSTICAS DEL CACHÉ
# ============================================================
stats_command() {
    echo ""
    echo "1️⃣  Ver estadísticas del caché"
    echo "════════════════════════════════════════════════════════════════"
    echo "Ejecutando: docker exec $INGESTOR_CONTAINER python cache_utils.py stats"
    docker exec $INGESTOR_CONTAINER python cache_utils.py stats
}

# ============================================================
# 2. INDEXAR NUEVOS PDFs
# ============================================================
index_command() {
    echo ""
    echo "2️⃣  Indexar nuevos PDFs (con caché automático)"
    echo "════════════════════════════════════════════════════════════════"
    echo "Los PDFs deben estar en subdirectorios de \$TOPIC_BASE_DIR"
    echo "Ejecutando: docker exec $INGESTOR_CONTAINER python main.py"
    docker exec $INGESTOR_CONTAINER python main.py
}

# ============================================================
# 3. EXPORTAR CACHÉ
# ============================================================
export_command() {
    local output_file="${1:-cache_backup_$(date +%Y%m%d_%H%M%S).json}"
    echo ""
    echo "3️⃣  Exportar caché a JSON"
    echo "════════════════════════════════════════════════════════════════"
    echo "Archivo de salida: $output_file"

    # Ejecutar export en contenedor
    docker exec $INGESTOR_CONTAINER python cache_utils.py export -o "/tmp/$output_file"

    # Copiar del contenedor a host
    docker cp "$INGESTOR_CONTAINER:/tmp/$output_file" "./$output_file"
    echo "✓ Caché exportado exitosamente"

    # Mostrar resumen
    echo ""
    echo "Resumen:"
    docker run --rm -v "$CACHE_VOLUME:/cache" \
        python:3.11-slim python3 -c \
        "import json; data = json.load(open('/cache/$output_file')); print(json.dumps(data['summary'], indent=2))" 2>/dev/null || echo "  (Usar jq para ver: jq .summary $output_file)"
}

# ============================================================
# 4. IMPORTAR CACHÉ
# ============================================================
import_command() {
    local input_file="$1"

    if [ -z "$input_file" ]; then
        echo "Error: Especificar archivo de entrada"
        echo "Uso: ./ejemplo_uso.sh import <archivo.json>"
        return 1
    fi

    if [ ! -f "$input_file" ]; then
        echo "Error: Archivo no encontrado: $input_file"
        return 1
    fi

    echo ""
    echo "4️⃣  Importar caché desde JSON"
    echo "════════════════════════════════════════════════════════════════"
    echo "Archivo de entrada: $input_file"

    # Copiar archivo al contenedor
    docker cp "$input_file" "$INGESTOR_CONTAINER:/tmp/$(basename $input_file)"

    # Ejecutar import
    docker exec $INGESTOR_CONTAINER python cache_utils.py import -i "/tmp/$(basename $input_file)"
    echo "✓ Caché importado exitosamente"
}

# ============================================================
# 5. LIMPIAR CACHÉ ANTIGUO
# ============================================================
clear_old_command() {
    local days="${1:-30}"
    echo ""
    echo "5️⃣  Limpiar caché más antiguo de $days días"
    echo "════════════════════════════════════════════════════════════════"

    read -p "¿Continuar? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        docker exec $INGESTOR_CONTAINER python cache_utils.py clear --days $days -y
        echo "✓ Limpieza completada"
    else
        echo "Cancelado"
    fi
}

# ============================================================
# 6. LIMPIAR TODO EL CACHÉ
# ============================================================
clear_all_command() {
    echo ""
    echo "6️⃣  LIMPIAR TODO EL CACHÉ"
    echo "════════════════════════════════════════════════════════════════"
    echo "⚠️  ADVERTENCIA: Se eliminará TODO el caché. No se puede deshacer."
    echo ""

    read -p "¿Continuar? Escribir 'sí' para confirmar: " -r
    if [[ $REPLY == "sí" ]]; then
        docker exec $INGESTOR_CONTAINER python cache_utils.py clear --all -y
        echo "✓ Caché completamente eliminado"
    else
        echo "Cancelado"
    fi
}

# ============================================================
# 7. OPTIMIZAR BASE DE DATOS
# ============================================================
vacuum_command() {
    echo ""
    echo "7️⃣  Optimizar base de datos SQLite"
    echo "════════════════════════════════════════════════════════════════"

    docker exec $INGESTOR_CONTAINER python cache_utils.py vacuum

    # Mostrar tamaño
    echo ""
    echo "Tamaño de caché:"
    docker exec $INGESTOR_CONTAINER du -sh /llava_cache/
}

# ============================================================
# 8. VER TAMAÑO DEL CACHÉ
# ============================================================
size_command() {
    echo ""
    echo "8️⃣  Ver tamaño del caché"
    echo "════════════════════════════════════════════════════════════════"

    docker exec $INGESTOR_CONTAINER du -sh /llava_cache/
    echo ""
    echo "Contenidos:"
    docker exec $INGESTOR_CONTAINER ls -lh /llava_cache/
}

# ============================================================
# 9. VER LOGS EN TIEMPO REAL
# ============================================================
logs_command() {
    echo ""
    echo "9️⃣  Ver logs en tiempo real"
    echo "════════════════════════════════════════════════════════════════"
    docker logs -f $INGESTOR_CONTAINER
}

# ============================================================
# 10. INFORMACIÓN DEL CACHÉ EN LA BD
# ============================================================
info_command() {
    echo ""
    echo "🔟  Información detallada de la BD"
    echo "════════════════════════════════════════════════════════════════"

    docker exec $INGESTOR_CONTAINER sqlite3 /llava_cache/llava_cache.db << EOF
.mode box
.headers on
.width 40 20 20

SELECT 'IMÁGENES EN CACHÉ' as tipo;
SELECT COUNT(*) as total_images,
       SUM(hit_count) as total_hits,
       ROUND(AVG(hit_count), 2) as avg_hits_per_image
FROM image_cache;

SELECT '' as '';

SELECT 'TABLAS EN CACHÉ' as tipo;
SELECT COUNT(*) as total_tables,
       SUM(hit_count) as total_hits,
       ROUND(AVG(hit_count), 2) as avg_hits_per_table
FROM table_cache;

SELECT '' as '';

SELECT 'TOP 5 IMÁGENES MÁS USADAS' as tipo;
SELECT image_hash, hit_count, created_at
FROM image_cache
ORDER BY hit_count DESC
LIMIT 5;

SELECT '' as '';

SELECT 'TOP 5 TABLAS MÁS USADAS' as tipo;
SELECT table_hash, hit_count, created_at
FROM table_cache
ORDER BY hit_count DESC
LIMIT 5;
EOF
}

# ============================================================
# 11. BACKUP AUTOMÁTICO
# ============================================================
backup_command() {
    echo ""
    echo "1️⃣1️⃣  Crear backup automático"
    echo "════════════════════════════════════════════════════════════════"

    local backup_dir="./backups"
    mkdir -p "$backup_dir"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_file="$backup_dir/cache_backup_$timestamp.json"

    docker exec $INGESTOR_CONTAINER python cache_utils.py export -o "/tmp/backup_$timestamp.json"
    docker cp "$INGESTOR_CONTAINER:/tmp/backup_$timestamp.json" "$backup_file"

    echo "✓ Backup creado: $backup_file"
    echo "  Tamaño: $(du -h $backup_file | cut -f1)"

    # Limpiar backups más antiguos de 7 días
    echo ""
    echo "Limpiando backups más antiguos de 7 días..."
    find "$backup_dir" -name "cache_backup_*.json" -mtime +7 -delete
}

# ============================================================
# 12. MONITOREO CONTINUO
# ============================================================
monitor_command() {
    echo ""
    echo "1️⃣2️⃣  Monitoreo continuo del caché"
    echo "════════════════════════════════════════════════════════════════"

    while true; do
        clear
        echo "╔════════════════════════════════════════════════════════════════╗"
        echo "║  MONITOREO EN TIEMPO REAL - $(date '+%Y-%m-%d %H:%M:%S')                  ║"
        echo "╚════════════════════════════════════════════════════════════════╝"

        echo ""
        echo "Tamaño de caché:"
        docker exec $INGESTOR_CONTAINER du -sh /llava_cache/

        echo ""
        echo "Estadísticas:"
        docker exec $INGESTOR_CONTAINER sqlite3 /llava_cache/llava_cache.db << EOF
SELECT 'Imágenes:' as '', COUNT(*) as cantidad, SUM(hit_count) as hits FROM image_cache
UNION ALL
SELECT 'Tablas:' as '', COUNT(*) as cantidad, SUM(hit_count) as hits FROM table_cache;
EOF

        echo ""
        echo "Actualizando cada 10 segundos (Ctrl+C para salir)..."
        sleep 10
    done
}

# ============================================================
# MENÚ PRINCIPAL
# ============================================================
show_menu() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  MENÚ PRINCIPAL - GESTIÓN DE CACHÉ SQLite                     ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  1) Ver estadísticas"
    echo "  2) Indexar PDFs"
    echo "  3) Exportar caché"
    echo "  4) Importar caché"
    echo "  5) Limpiar caché antiguo (30+ días)"
    echo "  6) Limpiar TODO el caché"
    echo "  7) Optimizar BD (VACUUM)"
    echo "  8) Ver tamaño"
    echo "  9) Ver logs"
    echo " 10) Información detallada"
    echo " 11) Backup automático"
    echo " 12) Monitoreo continuo"
    echo "  0) Salir"
    echo ""
}

# ============================================================
# MAIN
# ============================================================
if [ $# -eq 0 ]; then
    # Modo interactivo
    while true; do
        show_menu
        read -p "Seleccionar opción: " choice

        case $choice in
            1) stats_command ;;
            2) index_command ;;
            3) export_command ;;
            4) read -p "Archivo a importar: " file; import_command "$file" ;;
            5) read -p "Días (default 30): " days; clear_old_command "${days:-30}" ;;
            6) clear_all_command ;;
            7) vacuum_command ;;
            8) size_command ;;
            9) logs_command ;;
            10) info_command ;;
            11) backup_command ;;
            12) monitor_command ;;
            0) echo "Saliendo..."; exit 0 ;;
            *) echo "Opción inválida" ;;
        esac

        read -p "Presionar Enter para continuar..."
    done
else
    # Modo línea de comandos
    case "$1" in
        stats) stats_command ;;
        index) index_command ;;
        export) export_command "$2" ;;
        import) import_command "$2" ;;
        clear-old) clear_old_command "${2:-30}" ;;
        clear-all) clear_all_command ;;
        vacuum) vacuum_command ;;
        size) size_command ;;
        logs) logs_command ;;
        info) info_command ;;
        backup) backup_command ;;
        monitor) monitor_command ;;
        *)
            echo "Uso: $0 [comando] [opciones]"
            echo ""
            echo "Comandos:"
            echo "  stats              Ver estadísticas"
            echo "  index              Indexar PDFs"
            echo "  export [file]      Exportar caché"
            echo "  import <file>      Importar caché"
            echo "  clear-old [days]   Limpiar caché antiguo"
            echo "  clear-all          Limpiar TODO"
            echo "  vacuum             Optimizar BD"
            echo "  size               Ver tamaño"
            echo "  logs               Ver logs"
            echo "  info               Información detallada"
            echo "  backup             Backup automático"
            echo "  monitor            Monitoreo continuo"
            echo ""
            echo "Sin argumentos: Modo interactivo"
            exit 1
            ;;
    esac
fi
