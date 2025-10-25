DOCKER_USER="rristo"
IMAGE_NAME="et-tts-coqui"
IMAGE_TAG="0.1"

FULL_TAG="${DOCKER_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_TAG}"

docker push "${FULL_TAG}"
