#!/usr/bin/env bash
set -e

MODEL_ID="a8cheng/navila-llama3-8b-8f"
OUT_DIR="./checkpoints/navila-llama3-8b-8f"

hf download "$MODEL_ID" --local-dir "$OUT_DIR"

echo "Downloaded to: $OUT_DIR"