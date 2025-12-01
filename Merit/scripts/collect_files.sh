#!/bin/bash

# 출력 파일명
OUTPUT_FILE="collected_files.txt"

# 기존 파일이 있으면 삭제
rm -f "$OUTPUT_FILE"

echo "=== Python (.py) and YAML (.yaml) Files Collection ===" > "$OUTPUT_FILE"
echo "Generated on: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# .py 파일 찾기 및 처리
echo "Collecting .py files..." >&2
find . -name "*.py" -type f | while read -r file; do
    echo "================================" >> "$OUTPUT_FILE"
    echo "File: $file" >> "$OUTPUT_FILE"
    echo "================================" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

# .yaml 파일 찾기 및 처리
echo "Collecting .yaml files..." >&2
find . -name "*.yaml" -type f | while read -r file; do
    echo "================================" >> "$OUTPUT_FILE"
    echo "File: $file" >> "$OUTPUT_FILE"
    echo "================================" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
done

echo "Collection complete! Output saved to: $OUTPUT_FILE" >&2
echo "Total files processed:" >&2
find . -name "*.py" -o -name "*.yaml" | wc -l >&2
