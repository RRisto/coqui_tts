#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="coqui-et-tts"
IMAGE_TAG="0.1"

docker run --rm \
  --platform=linux/amd64 \
  -p 8080:8080 \
  -p 8265:8265 \
  --shm-size=1g \
  --ulimit nofile=65536:65536 \
  -e RAY_USAGE_STATS_ENABLED=0 \
  -e PORT=8080 \
  -e SERVER_TYPE=websocket \
  -v "$PWD/model":/app/model:ro \
  "${IMAGE_NAME}:${IMAGE_TAG}"