#!/bin/bash
# Run the linters. Use --fix to apply automatic fixes.

FIX_MODE=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fix) FIX_MODE=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ "$FIX_MODE" = true ]; then
    python -m ruff check --fix .
    python -m black .
else
    python -m ruff check .
    python -m black --check .
fi
