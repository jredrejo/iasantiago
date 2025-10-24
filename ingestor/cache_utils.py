#!/usr/bin/env python3
"""
Utilidades para gestionar el caché SQLite de LLaVA
Uso: python cache_utils.py [comando] [opciones]
"""

import sys
import argparse
import json
from pathlib import Path
from chunk import SQLiteCacheManager
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_size(size: int) -> str:
    """Convierte bytes a formato legible"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def stats_command(cache_db: str) -> None:
    """Muestra estadísticas del caché"""
    cache = SQLiteCacheManager(cache_db=cache_db)
    stats = cache.get_stats()
    top_items = cache.get_top_cached(limit=10)

    print("\n" + "=" * 70)
    print("CACHE STATISTICS")
    print("=" * 70)

    print(f"\nImágenes:")
    print(f"  Total en caché: {stats['images']['cached']}")
    print(f"  Total hits: {stats['images']['hits']}")
    if stats["images"]["cached"] > 0:
        avg_hits = stats["images"]["hits"] / stats["images"]["cached"]
        print(f"  Promedio de hits por imagen: {avg_hits:.1f}")

    print(f"\nTablas:")
    print(f"  Total en caché: {stats['tables']['cached']}")
    print(f"  Total hits: {stats['tables']['hits']}")
    if stats["tables"]["cached"] > 0:
        avg_hits = stats["tables"]["hits"] / stats["tables"]["cached"]
        print(f"  Promedio de hits por tabla: {avg_hits:.1f}")

    print(
        f"\nTotal en caché: {stats['images']['cached'] + stats['tables']['cached']} elementos"
    )
    print(f"Total hits: {stats['images']['hits'] + stats['tables']['hits']} accesos")

    if top_items["top_images"]:
        print(f"\nTop 10 imágenes más usadas:")
        for i, item in enumerate(top_items["top_images"], 1):
            print(
                f"  {i}. {item['image_hash']} - {item['hit_count']} hits - creada: {item['created_at']}"
            )

    if top_items["top_tables"]:
        print(f"\nTop 10 tablas más usadas:")
        for i, item in enumerate(top_items["top_tables"], 1):
            print(
                f"  {i}. {item['table_hash']} - {item['hit_count']} hits - creada: {item['created_at']}"
            )

    print("=" * 70 + "\n")


def clear_command(cache_db: str, days: int = None, confirm: bool = False) -> None:
    """Limpia el caché"""
    cache = SQLiteCacheManager(cache_db=cache_db)

    if days is None:
        # Limpiar TODO el caché
        if not confirm:
            response = input(f"¿Limpiar TODO el caché? (s/n): ").lower()
            if response != "s":
                print("Cancelado")
                return

        cache.clear_all_cache()
        print("✓ Caché completamente limpiado")
    else:
        # Limpiar antiguo
        if not confirm:
            response = input(
                f"¿Limpiar entradas más antiguas de {days} días? (s/n): "
            ).lower()
            if response != "s":
                print("Cancelado")
                return

        deleted = cache.clear_old_cache(days=days)
        print(f"✓ Limpiadas {deleted} entradas antiguas")


def export_command(cache_db: str, output_file: str) -> None:
    """Exporta caché a JSON"""
    import sqlite3

    try:
        conn = sqlite3.connect(cache_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Exportar imágenes
        cursor.execute("SELECT * FROM image_cache ORDER BY hit_count DESC")
        images = [dict(row) for row in cursor.fetchall()]

        # Exportar tablas
        cursor.execute("SELECT * FROM table_cache ORDER BY hit_count DESC")
        tables = [dict(row) for row in cursor.fetchall()]

        conn.close()

        data = {
            "images": images,
            "tables": tables,
            "summary": {
                "total_images": len(images),
                "total_tables": len(tables),
                "total_image_hits": sum(img["hit_count"] for img in images),
                "total_table_hits": sum(tbl["hit_count"] for tbl in tables),
            },
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✓ Caché exportado a {output_file}")
        print(f"  - {len(images)} imágenes")
        print(f"  - {len(tables)} tablas")

    except Exception as e:
        logger.error(f"Error exportando caché: {e}")


def import_command(cache_db: str, input_file: str) -> None:
    """Importa caché desde JSON"""
    import sqlite3

    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        cache = SQLiteCacheManager(cache_db=cache_db)

        # Importar imágenes
        for img in data.get("images", []):
            cache.save_image_cache(
                img["image_hash"], img["description"], img["width"], img["height"]
            )

        # Importar tablas
        for tbl in data.get("tables", []):
            cache.save_table_cache(
                tbl["table_hash"], tbl["analysis"], tbl["rows"], tbl["cols"]
            )

        print(f"✓ Caché importado desde {input_file}")
        print(f"  - {len(data.get('images', []))} imágenes importadas")
        print(f"  - {len(data.get('tables', []))} tablas importadas")

    except Exception as e:
        logger.error(f"Error importando caché: {e}")


def vacuum_command(cache_db: str) -> None:
    """Optimiza la base de datos SQLite"""
    import sqlite3

    try:
        conn = sqlite3.connect(cache_db)
        cursor = conn.cursor()

        # VACUUM comprime la BD
        cursor.execute("VACUUM")
        conn.commit()
        conn.close()

        db_path = Path(cache_db)
        size = db_path.stat().st_size
        print(f"✓ Base de datos optimizada")
        print(f"  Tamaño: {format_size(size)}")

    except Exception as e:
        logger.error(f"Error optimizando BD: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Herramientas para gestionar caché SQLite de LLaVA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s stats
  %(prog)s clear --all
  %(prog)s clear --days 30
  %(prog)s export -o cache_backup.json
  %(prog)s import -i cache_backup.json
  %(prog)s vacuum
        """,
    )

    parser.add_argument(
        "--db",
        default="/tmp/llava_cache/llava_cache.db",
        help="Ruta a la base de datos SQLite (default: %(default)s)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # stats
    subparsers.add_parser("stats", help="Mostrar estadísticas del caché")

    # clear
    clear_parser = subparsers.add_parser("clear", help="Limpiar caché")
    clear_group = clear_parser.add_mutually_exclusive_group()
    clear_group.add_argument("--all", action="store_true", help="Limpiar TODO el caché")
    clear_group.add_argument(
        "--days", type=int, help="Limpiar entradas más antiguas de N días"
    )
    clear_parser.add_argument(
        "-y", "--yes", action="store_true", help="No pedir confirmación"
    )

    # export
    export_parser = subparsers.add_parser("export", help="Exportar caché a JSON")
    export_parser.add_argument(
        "-o", "--output", required=True, help="Archivo de salida"
    )

    # import
    import_parser = subparsers.add_parser("import", help="Importar caché desde JSON")
    import_parser.add_argument(
        "-i", "--input", required=True, help="Archivo de entrada"
    )

    # vacuum
    subparsers.add_parser("vacuum", help="Optimizar base de datos")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "stats":
        stats_command(args.db)
    elif args.command == "clear":
        if args.all:
            clear_command(args.db, days=None, confirm=args.yes)
        elif args.days:
            clear_command(args.db, days=args.days, confirm=args.yes)
        else:
            parser.print_help()
    elif args.command == "export":
        export_command(args.db, args.output)
    elif args.command == "import":
        import_command(args.db, args.input)
    elif args.command == "vacuum":
        vacuum_command(args.db)


if __name__ == "__main__":
    main()
