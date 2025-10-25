#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="et-tts-coqui"
IMAGE_TAG="0.1"

docker run --rm \
  --platform=linux/amd64 \
  -p 9000:9000 \
  -p 8265:8265 \
  --shm-size=1g \
  --ulimit nofile=65536:65536 \
  -e RAY_USAGE_STATS_ENABLED=0 \
  -e PORT=9000 \
  -e SERVER_TYPE=grpc \
  -v "$PWD/model":/app/model:ro \
  "${IMAGE_NAME}:${IMAGE_TAG}"