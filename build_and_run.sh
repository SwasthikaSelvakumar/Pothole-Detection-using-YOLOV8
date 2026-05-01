#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  build_and_run.sh
#  Pothole Detection — Docker Build · Run · Push  |  Git Release
#  SDG 3 & SDG 11
# ════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Configuration ─────────────────────────────
DOCKERHUB_USER="${DOCKERHUB_USER:-venkatavamsi01}"
IMAGE_NAME="pothole-detector"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="pothole-detector-app"
HOST_PORT="${HOST_PORT:-5000}"

# ── Colours ───────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'
RED='\033[0;31m';   CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${CYAN}══════════════════════════════════${NC}"; \
          echo -e "${CYAN}  $*${NC}"; \
          echo -e "${CYAN}══════════════════════════════════${NC}"; }

# ── Usage ─────────────────────────────────────
usage() {
  echo ""
  echo "  Usage: $0 [command]"
  echo ""
  echo "  Commands:"
  echo "    build    Build the Docker image"
  echo "    run      Run the container locally"
  echo "    push     Push image to DockerHub"
  echo "    stop     Stop & remove container"
  echo "    logs     Tail container logs"
  echo "    tag      Git tag + push release"
  echo "    all      build + run + push"
  echo ""
  echo "  Environment variables:"
  echo "    GROQ_API_KEY    (required for run)"
  echo "    DOCKERHUB_USER  (default: venkatavamsi01)"
  echo "    HOST_PORT       (default: 5000)"
  echo ""
  exit 0
}

# ── BUILD ─────────────────────────────────────
build_image() {
  step "Building Docker Image: ${FULL_IMAGE}"

  if [ ! -f "best.pt" ]; then
    warn "YOLOv8 weights 'best.pt' not found in current directory."
    warn "Download from your training output or Roboflow and place it here."
    warn "Continuing build — container will fail at inference without it."
  fi

  docker build \
    --tag "${FULL_IMAGE}" \
    --label "build-date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --label "git-commit=$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
    --label "vcs-url=https://github.com/venkatavamsi01/Pothole-Detection-using-YOLOV8" \
    .

  info "Image built → ${FULL_IMAGE}"
}

# ── RUN ───────────────────────────────────────
run_container() {
  step "Starting Container: ${CONTAINER_NAME}"

  # Stop existing
  if docker ps -aq --filter "name=${CONTAINER_NAME}" | grep -q .; then
    warn "Removing existing container …"
    docker rm -f "${CONTAINER_NAME}"
  fi

  # Check API key
  if [ -z "${GROQ_API_KEY:-}" ]; then
    error "GROQ_API_KEY is not set.\n  Run: export GROQ_API_KEY=your_key_here"
  fi

  docker run \
    --detach \
    --name    "${CONTAINER_NAME}" \
    --publish "${HOST_PORT}:5000" \
    --env     "GROQ_API_KEY=${GROQ_API_KEY}" \
    --env     "MODEL_PATH=/app/best.pt" \
    --restart unless-stopped \
    "${FULL_IMAGE}"

  info "Container is running!"
  echo ""
  echo "  Test endpoints:"
  echo "    Health  →  curl http://localhost:${HOST_PORT}/health"
  echo "    Classes →  curl http://localhost:${HOST_PORT}/classes"
  echo "    Predict →  curl -X POST http://localhost:${HOST_PORT}/predict \\"
  echo "                    -F 'image=@/path/to/road.jpg'"
  echo ""
}

# ── PUSH ──────────────────────────────────────
push_image() {
  step "Pushing to DockerHub: ${FULL_IMAGE}"
  docker login --username "${DOCKERHUB_USER}"
  docker push "${FULL_IMAGE}"
  info "Image live at: https://hub.docker.com/r/${DOCKERHUB_USER}/${IMAGE_NAME}"
}

# ── STOP ──────────────────────────────────────
stop_container() {
  step "Stopping Container: ${CONTAINER_NAME}"
  if docker ps -q --filter "name=${CONTAINER_NAME}" | grep -q .; then
    docker rm -f "${CONTAINER_NAME}"
    info "Container stopped and removed."
  else
    warn "No running container '${CONTAINER_NAME}' found."
  fi
}

# ── LOGS ──────────────────────────────────────
show_logs() {
  docker logs -f "${CONTAINER_NAME}"
}

# ── GIT TAG ───────────────────────────────────
git_tag_release() {
  step "Git Release Tagging"
  TAG="v1.0.0"

  git add -A
  git commit -m "feat: YOLOv8 pothole detection + Groq LLM agentic loop + Docker" \
    || warn "Nothing new to commit."

  git tag -a "${TAG}" -m "Release ${TAG} — YOLOv8 + Groq LLM + Docker | SDG 3 & 11"
  git push origin main --tags

  info "Release ${TAG} pushed to GitHub."
}

# ── ENTRY POINT ───────────────────────────────
COMMAND="${1:-usage}"

case "${COMMAND}" in
  build)  build_image ;;
  run)    run_container ;;
  push)   push_image ;;
  stop)   stop_container ;;
  logs)   show_logs ;;
  tag)    git_tag_release ;;
  all)
    build_image
    run_container
    push_image
    ;;
  *) usage ;;
esac
