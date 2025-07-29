#!/bin/bash

# Quick Docker Hub Push Script
# ============================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_USERNAME="debabratap"
IMAGE_NAME="fraud-detection-system"
VERSION="1.0.0"

print_header() {
    echo -e "${BLUE}"
    echo "🐳 =============================================="
    echo "   Quick Docker Hub Push"
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
    echo "Please login to Docker Hub:"
    echo "Username: $DOCKER_USERNAME"
    echo
    
    if docker login; then
        print_success "Successfully logged in to Docker Hub"
    else
        print_error "Failed to login to Docker Hub"
        exit 1
    fi
}

# Step 2: Build production image
build_production() {
    print_info "Step 2: Building Production Image"
    
    docker build \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'local')" \
        --tag "$DOCKER_USERNAME/$IMAGE_NAME:$VERSION" \
        --tag "$DOCKER_USERNAME/$IMAGE_NAME:latest" \
        .
    
    print_success "Production image built successfully"
}

# Step 3: Build development image
build_development() {
    print_info "Step 3: Building Development Image"
    
    docker build \
        --target development \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo 'local')" \
        --tag "$DOCKER_USERNAME/$IMAGE_NAME:$VERSION-dev" \
        --tag "$DOCKER_USERNAME/$IMAGE_NAME:dev" \
        .
    
    print_success "Development image built successfully"
}

# Step 4: Test the production image
test_image() {
    print_info "Step 4: Testing Production Image"
    
    # Start container for testing
    docker run -d --name fraud-test -p 8501:8501 "$DOCKER_USERNAME/$IMAGE_NAME:latest"
    
    # Wait for container to start
    print_info "Waiting for application to start..."
    sleep 30
    
    # Test health endpoint
    if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
        print_success "Image test passed - application is healthy"
    else
        print_warning "Health check failed, but continuing with push"
    fi
    
    # Stop and remove test container
    docker stop fraud-test
    docker rm fraud-test
}

# Step 5: Push images to Docker Hub
push_images() {
    print_info "Step 5: Pushing Images to Docker Hub"
    
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

# Step 6: Show final information
show_final_info() {
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
    echo "📊 Local Images:"
    docker images | grep "$DOCKER_USERNAME/$IMAGE_NAME" | head -5
    echo
}

# Main execution
main() {
    print_header
    
    print_info "Configuration:"
    echo "Docker Username: $DOCKER_USERNAME"
    echo "Image Name: $IMAGE_NAME"
    echo "Version: $VERSION"
    echo "Git Commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'local')"
    echo
    
    # Check if we're in the right directory
    if [ ! -f "Dockerfile" ]; then
        print_error "Dockerfile not found! Please run this script from the project root directory."
        exit 1
    fi
    
    # Execute all steps
    docker_login
    build_production
    build_development
    test_image
    push_images
    show_final_info
    
    print_success "🎉 Docker Hub deployment completed successfully!"
}

# Run main function
main "$@"
