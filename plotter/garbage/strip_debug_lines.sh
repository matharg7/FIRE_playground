#!/bin/bash
# Removes lines starting with "n_drop_per_tile" or "n_ones actual" from .out log files.
# Usage:
#   ./strip_debug_lines.sh <file_or_directory>
#
# If given a file, it strips that file in-place.
# If given a directory, it strips all .out files in that directory.

set -euo pipefail

strip_file() {
    local file="$1"
    local original_lines new_lines

    original_lines=$(wc -l < "$file")
    
    # Use grep to remove matching lines, writing to a temp file then replacing
    local tmpfile="${file}.tmp"
    grep -v -E '^(n_drop_per_tile|n_ones actual)' "$file" > "$tmpfile" || true
    mv "$tmpfile" "$file"

    new_lines=$(wc -l < "$file")
    local removed=$((original_lines - new_lines))
    echo "  $file: ${original_lines} -> ${new_lines} lines (removed ${removed})"
}

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file_or_directory>"
    exit 1
fi

target="$1"

if [ -f "$target" ]; then
    echo "Stripping debug lines from file:"
    strip_file "$target"
elif [ -d "$target" ]; then
    echo "Stripping debug lines from all .out files in: $target"
    found=0
    for f in "$target"/*.out; do
        [ -f "$f" ] || continue
        strip_file "$f"
        found=$((found + 1))
    done
    if [ "$found" -eq 0 ]; then
        echo "  No .out files found."
    else
        echo "Done. Processed $found file(s)."
    fi
else
    echo "Error: '$target' is not a file or directory."
    exit 1
fi
