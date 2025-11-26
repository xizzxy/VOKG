#!/bin/bash
set -euo pipefail

# VOKG Production Deployment Script
# Usage: ./deploy.sh <environment> <version>
#
# Examples:
#   ./deploy.sh staging v1.2.3
#   ./deploy.sh production v1.2.3

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    log_error "Environment must be 'staging' or 'production'"
    exit 1
fi

# Confirm production deployment
if [[ "$ENVIRONMENT" == "production" ]]; then
    log_warn "You are about to deploy to PRODUCTION!"
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        log_info "Deployment cancelled"
        exit 0
    fi
fi

log_info "Starting deployment to $ENVIRONMENT with version $VERSION"

# Check prerequisites
log_info "Checking prerequisites..."
command -v kubectl >/dev/null 2>&1 || { log_error "kubectl is required but not installed"; exit 1; }
command -v helm >/dev/null 2>&1 || { log_error "helm is required but not installed"; exit 1; }

# Set kubectl context
log_info "Setting kubectl context for $ENVIRONMENT..."
if [[ "$ENVIRONMENT" == "production" ]]; then
    kubectl config use-context vokg-production
else
    kubectl config use-context vokg-staging
fi

# Verify context
CURRENT_CONTEXT=$(kubectl config current-context)
log_info "Current context: $CURRENT_CONTEXT"

# Create namespace if it doesn't exist
log_info "Ensuring namespace exists..."
kubectl apply -f "$PROJECT_ROOT/infrastructure/kubernetes/namespace.yaml"

# Apply secrets (ensure these are set externally via secret management)
log_info "Verifying secrets..."
if ! kubectl get secret vokg-secrets -n vokg >/dev/null 2>&1; then
    log_error "Secret 'vokg-secrets' not found. Please configure secrets first."
    exit 1
fi

# Export variables for envsubst
export REGISTRY="ghcr.io/your-org"
export VERSION

# Deploy API Gateway
log_info "Deploying API Gateway..."
envsubst < "$PROJECT_ROOT/infrastructure/kubernetes/deployment-api.yaml" | kubectl apply -f -

# Deploy CPU Workers
log_info "Deploying CPU Workers..."
envsubst < "$PROJECT_ROOT/infrastructure/kubernetes/deployment-worker-cpu.yaml" | kubectl apply -f -

# Deploy GPU Workers
log_info "Deploying GPU Workers..."
envsubst < "$PROJECT_ROOT/infrastructure/kubernetes/deployment-worker-gpu.yaml" | kubectl apply -f -

# Deploy LLM Workers
log_info "Deploying LLM Workers..."
envsubst < "$PROJECT_ROOT/infrastructure/kubernetes/deployment-worker-llm.yaml" | kubectl apply -f -

# Deploy Ingress
log_info "Deploying Ingress..."
envsubst < "$PROJECT_ROOT/infrastructure/kubernetes/ingress.yaml" | kubectl apply -f -

# Wait for rollouts
log_info "Waiting for API rollout to complete..."
kubectl rollout status deployment/vokg-api -n vokg --timeout=10m

log_info "Waiting for CPU worker rollout to complete..."
kubectl rollout status deployment/vokg-worker-cpu -n vokg --timeout=10m

log_info "Waiting for GPU worker rollout to complete..."
kubectl rollout status deployment/vokg-worker-gpu -n vokg --timeout=10m

log_info "Waiting for LLM worker rollout to complete..."
kubectl rollout status deployment/vokg-worker-llm -n vokg --timeout=10m

# Verify deployment
log_info "Verifying deployment..."
kubectl get pods -n vokg -l app=vokg

# Run smoke tests
log_info "Running smoke tests..."
if [[ "$ENVIRONMENT" == "production" ]]; then
    HEALTH_URL="https://vokg.example.com/api/health"
else
    HEALTH_URL="https://staging.vokg.example.com/api/health"
fi

if curl -f -s "$HEALTH_URL" > /dev/null; then
    log_info "Health check passed"
else
    log_error "Health check failed"
    exit 1
fi

# Display deployment info
log_info "Deployment Summary:"
echo "  Environment: $ENVIRONMENT"
echo "  Version: $VERSION"
echo "  Context: $CURRENT_CONTEXT"
echo ""
kubectl get deployments -n vokg
echo ""
kubectl get pods -n vokg

log_info "Deployment completed successfully!"

# Remind about monitoring
log_info "Don't forget to monitor:"
echo "  - Grafana: https://grafana.vokg.example.com"
echo "  - Prometheus: https://prometheus.vokg.example.com"
echo "  - Logs: kubectl logs -f -n vokg -l app=vokg"
