#!/usr/bin/env bash
set -euo pipefail

# Configuration
IMAGE_NAME="coqui-et-tts"
IMAGE_TAG="latest"
DOCKERFILE="Dockerfile"
DEFAULT_PLATFORM="linux/amd64"  # Ray base image with Python 3.11 (CPU optimized)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -n, --name NAME     Docker image name (default: coqui-et-tts)"
    echo "  -t, --tag TAG       Docker image tag (default: latest)"
    echo "  -f, --file FILE     Dockerfile path (default: Dockerfile)"
    echo "  --no-cache          Build without using cache"
    echo "  --platform PLATFORM Build for specific platform (default: linux/amd64, Ray base image with Python 3.11 and CUDA support)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build with defaults (linux/amd64)"
    echo "  $0 -n my-tts -t v1.0                 # Custom name and tag"
    echo "  $0 --no-cache                        # Build without cache (linux/amd64)"
    echo "  $0 --platform linux/amd64            # Explicitly specify platform"
}

# Parse command line arguments
BUILD_ARGS=()
PLATFORM=""
NO_CACHE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -f|--file)
            DOCKERFILE="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! -f "$DOCKERFILE" ]]; then
    print_error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

if [[ ! -d "model" ]]; then
    print_error "Model directory not found. Please ensure the 'model' directory exists."
    exit 1
fi

# Build the Docker image
print_status "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
print_status "Using Dockerfile: $DOCKERFILE"

# Use default platform if none specified (Ray base image is amd64 only)
if [[ -z "$PLATFORM" ]]; then
    PLATFORM="$DEFAULT_PLATFORM"
    print_warning "No platform specified, using default: $PLATFORM (Ray base image with Python 3.11 and CUDA support)"
fi

BUILD_CMD="docker buildx build --platform $PLATFORM"
print_status "Building for platform: $PLATFORM"

if [[ -n "$NO_CACHE" ]]; then
    BUILD_CMD="$BUILD_CMD $NO_CACHE"
    print_status "Building without cache"
fi

BUILD_CMD="$BUILD_CMD -f $DOCKERFILE -t ${IMAGE_NAME}:${IMAGE_TAG} ."

print_status "Executing: $BUILD_CMD"
eval $BUILD_CMD

if [[ $? -eq 0 ]]; then
    print_status "Docker image built successfully!"
    print_status "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    print_status "To run the container:"
    echo "  docker run -p 8080:8080 ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    print_status "To run with custom port:"
    echo "  docker run -p 9000:9000 -e PORT=9000 ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    print_status "To run with gRPC mode:"
    echo "  docker run -p 8080:8080 -e SERVER_TYPE=grpc ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    print_status "To run with WebSocket mode (default):"
    echo "  docker run -p 8080:8080 -e SERVER_TYPE=websocket ${IMAGE_NAME}:${IMAGE_TAG}"
else
    print_error "Docker build failed!"
    exit 1
fi