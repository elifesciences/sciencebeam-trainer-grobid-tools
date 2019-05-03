#!/bin/bash

set -e

source .config

source_base_path="$GENERATED_TRAINING_DATA_DIR"
output_path="$AUTO_GENERATED_GENERATED_TRAINING_DATA_DIR"

python -m grobid_training.auto_annotate_header \
    --source-base-path "$source_base_path" \
    --output-path "$output_path" \
    --xml-path "$XML_PATH" \
    --xml-filename-regex "$XML_FILENAME_REGEX" \
    --fields manuscript_title \
    $@
