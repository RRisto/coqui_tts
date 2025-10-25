#!/usr/bin/env bash
set -euo pipefail

# Name and tag for your image
IMAGE_NAME=et-tts-coqui
IMAGE_TAG=0.1

# Build for amd64 (most Anyscale clusters use amd64)
docker buildx build \
  --platform=linux/amd64 \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  --load .


