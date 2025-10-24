#!/bin/bash

# manage_state.sh - Herramienta para gestionar el estado de procesamiento del ingestor

set -e

CONTAINER="ingestor"
STATE_FILE="/whoosh/.processing_state.json"

# ============================================================
# COLORS for output
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# FUNCTIONS
# ============================================================

show_help() {
    cat << EOF
Usage: $(basename "$0") [COMMAND]

Commands:
  status          Show current processing state and statistics
  list-processed  List all processed files
  list-failed     List all failed files
  reset           Reset state (force complete rescan)
  rescan FILE     Rescan a specific file
  clean           Delete and recreate state file
  backup          Backup current state
  restore FILE    Restore state from backup
  validate        Validate state file integrity
  help            Show this help message

Examples:
  $(basename "$0") status
  $(basename "$0") list-failed
  $(basename "$0") rescan /topics/Chemistry/tema_1.pdf
  $(basename "$0") reset
EOF
}

check_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        echo -e "${RED}✗ Container '$CONTAINER' is not running${NC}"
        exit 1
    fi
}

show_status() {
    check_container
    
    echo -e "${BLUE}Processing State Status${NC}"
    echo "======================================"
    
    docker exec "$CONTAINER" python -c "
import json
from datetime import datetime

try:
    with open('$STATE_FILE') as f:
        state = json.load(f)
except FileNotFoundError:
    print('✗ State file not found')
    exit(1)
except json.JSONDecodeError:
    print('✗ State file corrupted')
    exit(1)

processed = state.get('processed', {})
successful = sum(1 for v in processed.values() if v.get('status') == 'success')
failed = sum(1 for v in processed.values() if v.get('status') == 'failed')

print(f'Total Processed: {len(processed)}')
print(f'  ✓ Successful: {successful}')
print(f'  ✗ Failed: {failed}')
print(f'Last Scan: {state.get(\"last_scan\", \"Never\")}')
print(f'Created At: {state.get(\"created_at\", \"Unknown\")}')

if failed > 0:
    print(f'\n{failed} files failed:')
    for path, info in processed.items():
        if info.get('status') == 'failed':
            error = info.get('error', 'Unknown')[:60]
            print(f'  ✗ {path}')
            print(f'    → {error}...')
" || exit 1
}

list_processed() {
    check_container
    
    echo -e "${BLUE}Processed Files${NC}"
    echo "======================================"
    
    docker exec "$CONTAINER" python -c "
import json

with open('$STATE_FILE') as f:
    state = json.load(f)

processed = state.get('processed', {})
successful = [k for k, v in processed.items() if v.get('status') == 'success']

print(f'Total: {len(successful)} files\n')
for path in sorted(successful):
    info = processed[path]
    print(f'✓ {path}')
    print(f'  Topic: {info.get(\"topic\", \"Unknown\")}')
    print(f'  Time: {info.get(\"timestamp\", \"Unknown\")}')
"
}

list_failed() {
    check_container
    
    echo -e "${RED}Failed Files${NC}"
    echo "======================================"
    
    docker exec "$CONTAINER" python -c "
import json

with open('$STATE_FILE') as f:
    state = json.load(f)

failed_dict = state.get('failed', {})

if not failed_dict:
    print('No failed files')
    exit(0)

print(f'Total: {len(failed_dict)} files\n')
for path, info in sorted(failed_dict.items()):
    print(f'✗ {path}')
    error = info.get('error', 'Unknown')
    # Show first 200 chars of error
    print(f'  Error: {error[:200]}')
    print(f'  Time: {info.get(\"timestamp\", \"Unknown\")}')
"
}

reset_state() {
    check_container
    
    echo -e "${YELLOW}⚠️  Resetting processing state${NC}"
    echo "This will force a complete rescan of all PDFs"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled"
        exit 0
    fi
    
    docker exec "$CONTAINER" python -c "
import json
from datetime import datetime

state = {
    'version': 1,
    'created_at': datetime.now().isoformat(),
    'last_scan': None,
    'processed': {},
    'failed': {}
}

with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)

print('✓ State reset')
" || exit 1
    
    echo -e "${GREEN}✓ State has been reset${NC}"
    echo "Next scan will process all files"
}

rescan_file() {
    check_container
    
    if [ -z "$1" ]; then
        echo -e "${RED}✗ Please provide file path${NC}"
        exit 1
    fi
    
    FILE_PATH="$1"
    
    echo "Removing '$FILE_PATH' from state..."
    
    docker exec "$CONTAINER" python -c "
import json

with open('$STATE_FILE') as f:
    state = json.load(f)

if '$FILE_PATH' in state['processed']:
    del state['processed']['$FILE_PATH']
    print('✓ Removed from processed')

if '$FILE_PATH' in state['failed']:
    del state['failed']['$FILE_PATH']
    print('✓ Removed from failed')

with open('$STATE_FILE', 'w') as f:
    json.dump(state, f, indent=2)

print('File will be rescanned on next run')
" || exit 1
}

clean_state() {
    check_container
    
    echo -e "${YELLOW}⚠️  Deleting state file${NC}"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled"
        exit 0
    fi
    
    docker exec "$CONTAINER" rm "$STATE_FILE"
    echo -e "${GREEN}✓ State file deleted${NC}"
}

backup_state() {
    check_container
    
    BACKUP_FILE=".processing_state.backup.$(date +%Y%m%d_%H%M%S).json"
    
    echo "Creating backup: $BACKUP_FILE"
    docker exec "$CONTAINER" cp "$STATE_FILE" "/whoosh/$BACKUP_FILE"
    echo -e "${GREEN}✓ Backup created: /whoosh/$BACKUP_FILE${NC}"
}

restore_state() {
    check_container
    
    if [ -z "$1" ]; then
        echo -e "${RED}✗ Please provide backup file name${NC}"
        echo "Available backups:"
        docker exec "$CONTAINER" ls -la /whoosh/.processing_state.backup.* 2>/dev/null || echo "No backups found"
        exit 1
    fi
    
    BACKUP_FILE="/whoosh/$1"
    
    echo -e "${YELLOW}⚠️  Restoring state from $1${NC}"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled"
        exit 0
    fi
    
    docker exec "$CONTAINER" cp "$BACKUP_FILE" "$STATE_FILE"
    echo -e "${GREEN}✓ State restored${NC}"
}

validate_state() {
    check_container
    
    echo -e "${BLUE}Validating state file...${NC}"
    
    docker exec "$CONTAINER" python -c "
import json

try:
    with open('$STATE_FILE') as f:
        state = json.load(f)
    
    # Basic validation
    assert 'version' in state
    assert 'processed' in state
    assert 'failed' in state
    
    print('✓ State file is valid')
    print(f'  Version: {state[\"version\"]}')
    print(f'  Processed entries: {len(state[\"processed\"])}')
    print(f'  Failed entries: {len(state[\"failed\"])}')
    
except FileNotFoundError:
    print('✗ State file not found')
    exit(1)
except json.JSONDecodeError as e:
    print(f'✗ State file is corrupted: {e}')
    exit(1)
except AssertionError:
    print('✗ State file format is invalid')
    exit(1)
"
}

# ============================================================
# MAIN
# ============================================================

COMMAND="${1:-help}"

case "$COMMAND" in
    status)
        show_status
        ;;
    list-processed)
        list_processed
        ;;
    list-failed)
        list_failed
        ;;
    reset)
        reset_state
        ;;
    rescan)
        rescan_file "$2"
        ;;
    clean)
        clean_state
        ;;
    backup)
        backup_state
        ;;
    restore)
        restore_state "$2"
        ;;
    validate)
        validate_state
        ;;
    help)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac
