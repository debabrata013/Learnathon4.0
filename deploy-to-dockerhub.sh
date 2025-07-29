#!/bin/bash

# Auto Insurance Fraud Detection System - Docker Hub Deployment
# =============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_USERNAME="debabratapattnayak"
IMAGE_NAME="fraud-detection-system"
VERSION="1.0.0"

print_header() {
    echo -e "${BLUE}"
    echo "🐳 =============================================="
    echo "   Docker Hub Deployment Script"
    echo "   Auto Insurance Fraud Detection System"
    echo "===============================================${NC}"
    echo
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Step 1: Login to Docker Hub
docker_login() {
    print_info "Step 1: Docker Hub Login"
    echo "Please login to Docker Hub with your credentials:"
    echo "Username: $DOCKER_USERNAME"
    echo
    
    if docker login; then
        print_success "Successfully logged in to Docker Hub"
    else
        print_error "Failed to login to Docker Hub"
        exit 1
    fi
}

# Step 2: Build multi-stage images
build_images() {
    print_info "Step 2: Building Multi-Stage Docker Images"
    
    # Build production image
    print_info "Building production image..."
    docker build \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --tag "$DOCKER_USERNAME/$IMAGE_NAME:$VERSION" \
        --tag "$DOCKER_USERNAME/$IMAGE_NAME:latest" \
        .
    
    print_success "Production image built successfully"
    
    # Build development image
    print_info "Building development image..."
    docker build \
        --target development \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
        --tag "$DOCKER_USERNAME/$IMAGE_NAME:$VERSION-dev" \
        --tag "$DOCKER_USERNAME/$IMAGE_NAME:dev" \
        .
    
    print_success "Development image built successfully"
}

# Step 3: Test images
test_images() {
    print_info "Step 3: Testing Built Images"
    
    # Test production image
    print_info "Testing production image..."
    
    # Start container
    docker run -d --name fraud-test-prod -p 8501:8501 "$DOCKER_USERNAME/$IMAGE_NAME:latest"
    
    # Wait for startup
    print_info "Waiting for application to start..."
    sleep 30
    
    # Test health endpoint
    if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
        print_success "Production image health check passed"
    else
        print_warning "Health check failed, but continuing..."
    fi
    
    # Stop and remove test container
    docker stop fraud-test-prod
    docker rm fraud-test-prod
    
    print_success "Image testing completed"
}

# Step 4: Push to Docker Hub
push_images() {
    print_info "Step 4: Pushing Images to Docker Hub"
    
    # Push production images
    print_info "Pushing production images..."
    docker push "$DOCKER_USERNAME/$IMAGE_NAME:$VERSION"
    docker push "$DOCKER_USERNAME/$IMAGE_NAME:latest"
    
    # Push development images
    print_info "Pushing development images..."
    docker push "$DOCKER_USERNAME/$IMAGE_NAME:$VERSION-dev"
    docker push "$DOCKER_USERNAME/$IMAGE_NAME:dev"
    
    print_success "All images pushed successfully to Docker Hub!"
}

# Step 5: Verify deployment
verify_deployment() {
    print_info "Step 5: Verifying Deployment"
    
    # Pull and test the pushed image
    print_info "Pulling image from Docker Hub..."
    docker pull "$DOCKER_USERNAME/$IMAGE_NAME:latest"
    
    # Quick test
    print_info "Testing pulled image..."
    docker run --rm --name verify-test -d -p 8502:8501 "$DOCKER_USERNAME/$IMAGE_NAME:latest"
    sleep 20
    
    if curl -f http://localhost:8502/_stcore/health &>/dev/null; then
        print_success "Deployed image verification successful"
    else
        print_warning "Verification test failed"
    fi
    
    docker stop verify-test 2>/dev/null || true
    
    print_success "Deployment verification completed"
}

# Step 6: Show deployment information
show_deployment_info() {
    print_info "Step 6: Deployment Information"
    echo
    echo "🎉 Your Docker images are now available on Docker Hub!"
    echo
    echo "📦 Available Images:"
    echo "   Production (Latest): docker pull $DOCKER_USERNAME/$IMAGE_NAME:latest"
    echo "   Production (v$VERSION): docker pull $DOCKER_USERNAME/$IMAGE_NAME:$VERSION"
    echo "   Development (Latest): docker pull $DOCKER_USERNAME/$IMAGE_NAME:dev"
    echo "   Development (v$VERSION): docker pull $DOCKER_USERNAME/$IMAGE_NAME:$VERSION-dev"
    echo
    echo "🚀 Quick Start Commands:"
    echo "   docker run -p 8501:8501 $DOCKER_USERNAME/$IMAGE_NAME:latest"
    echo "   docker run -p 8501:8501 -p 8888:8888 $DOCKER_USERNAME/$IMAGE_NAME:dev"
    echo
    echo "🌐 Docker Hub Repository:"
    echo "   https://hub.docker.com/r/$DOCKER_USERNAME/$IMAGE_NAME"
    echo
    echo "📊 Image Information:"
    docker images | grep "$DOCKER_USERNAME/$IMAGE_NAME" | head -5
    echo
}

# Cleanup function
cleanup() {
    print_info "Cleaning up test containers..."
    docker stop fraud-test-prod verify-test 2>/dev/null || true
    docker rm fraud-test-prod verify-test 2>/dev/null || true
}

# Set up cleanup trap
trap cleanup EXIT

# Main execution
main() {
    print_header
    
    print_info "Deployment Configuration:"
    echo "Docker Username: $DOCKER_USERNAME"
    echo "Image Name: $IMAGE_NAME"
    echo "Version: $VERSION"
    echo "Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo
    
    # Check prerequisites
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    # Execute deployment steps
    docker_login
    build_images
    test_images
    push_images
    verify_deployment
    show_deployment_info
    
    print_success "🎉 Docker Hub deployment completed successfully!"
    
    # Ask if user wants to clean up local images
    echo
    read -p "Do you want to clean up local build images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning up local images..."
        docker image prune -f
        print_success "Local cleanup completed"
    fi
}

# Run main function
main "$@"
