#!/bin/bash
set -euo pipefail

# VOKG Worker Scaling Script
# Usage: ./scale-workers.sh <worker-type> <replica-count>
#
# Examples:
#   ./scale-workers.sh cpu 10
#   ./scale-workers.sh gpu 3
#   ./scale-workers.sh llm 5

WORKER_TYPE=${1:-}
REPLICA_COUNT=${2:-}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate inputs
if [[ -z "$WORKER_TYPE" || -z "$REPLICA_COUNT" ]]; then
    log_error "Usage: $0 <worker-type> <replica-count>"
    echo "Worker types: cpu, gpu, llm"
    exit 1
fi

if [[ ! "$WORKER_TYPE" =~ ^(cpu|gpu|llm)$ ]]; then
    log_error "Invalid worker type. Must be cpu, gpu, or llm"
    exit 1
fi

if ! [[ "$REPLICA_COUNT" =~ ^[0-9]+$ ]]; then
    log_error "Replica count must be a number"
    exit 1
fi

# Warn about GPU scaling costs
if [[ "$WORKER_TYPE" == "gpu" && "$REPLICA_COUNT" -gt 3 ]]; then
    log_warn "WARNING: GPU workers are expensive!"
    log_warn "Current cost: ~\$0.526/hour per g4dn.xlarge instance"
    log_warn "Scaling to $REPLICA_COUNT replicas will cost ~\$$(echo "$REPLICA_COUNT * 0.526" | bc)/hour"
    read -p "Continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        log_info "Scaling cancelled"
        exit 0
    fi
fi

DEPLOYMENT_NAME="vokg-worker-$WORKER_TYPE"

log_info "Scaling $DEPLOYMENT_NAME to $REPLICA_COUNT replicas..."

# Scale the deployment
kubectl scale deployment "$DEPLOYMENT_NAME" -n vokg --replicas="$REPLICA_COUNT"

# Wait for rollout
log_info "Waiting for scaling to complete..."
kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n vokg --timeout=5m

# Display current status
log_info "Current worker status:"
kubectl get pods -n vokg -l component=worker-"$WORKER_TYPE"

# Show queue status
log_info "Current queue lengths:"
if command -v redis-cli >/dev/null 2>&1; then
    REDIS_POD=$(kubectl get pods -n vokg -l app=redis -o jsonpath='{.items[0].metadata.name}')
    if [[ -n "$REDIS_POD" ]]; then
        for queue in cpu gpu llm graph; do
            LENGTH=$(kubectl exec -n vokg "$REDIS_POD" -- redis-cli LLEN "celery:$queue" 2>/dev/null || echo "N/A")
            echo "  $queue: $LENGTH"
        done
    fi
fi

log_info "Scaling completed successfully!"
