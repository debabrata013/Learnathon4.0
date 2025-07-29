#!/bin/bash

# Auto Insurance Fraud Detection System - Docker Build & Push Script
# ===================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-debabratapattnayak}"
IMAGE_NAME="fraud-detection-system"
VERSION="${VERSION:-1.0.0}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Full image names
PRODUCTION_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
PRODUCTION_LATEST="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
DEVELOPMENT_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-dev"
DEVELOPMENT_LATEST="${DOCKER_USERNAME}/${IMAGE_NAME}:dev"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "🐳 =============================================="
    echo "   Docker Build & Push Script"
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

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if logged in to Docker Hub
    if ! docker info | grep -q "Username:"; then
        print_warning "Not logged in to Docker Hub"
        print_info "Please run: docker login"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    print_success "Prerequisites check passed"
}

# Build production image
build_production() {
    print_info "Building production image..."
    
    docker build \
        --target production \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$VCS_REF" \
        --tag "$PRODUCTION_IMAGE" \
        --tag "$PRODUCTION_LATEST" \
        .
    
    print_success "Production image built: $PRODUCTION_IMAGE"
}

# Build development image
build_development() {
    print_info "Building development image..."
    
    docker build \
        --target development \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$VERSION" \
        --build-arg VCS_REF="$VCS_REF" \
        --tag "$DEVELOPMENT_IMAGE" \
        --tag "$DEVELOPMENT_LATEST" \
        .
    
    print_success "Development image built: $DEVELOPMENT_IMAGE"
}

# Test images
test_images() {
    print_info "Testing built images..."
    
    # Test production image
    print_info "Testing production image..."
    docker run --rm --name fraud-test-prod -d -p 8501:8501 "$PRODUCTION_IMAGE"
    
    # Wait for container to start
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:8501/_stcore/health &>/dev/null; then
        print_success "Production image health check passed"
    else
        print_warning "Production image health check failed"
    fi
    
    # Stop test container
    docker stop fraud-test-prod
    
    print_success "Image testing completed"
}

# Push images to Docker Hub
push_images() {
    print_info "Pushing images to Docker Hub..."
    
    # Push production images
    print_info "Pushing production images..."
    docker push "$PRODUCTION_IMAGE"
    docker push "$PRODUCTION_LATEST"
    
    # Push development images
    print_info "Pushing development images..."
    docker push "$DEVELOPMENT_IMAGE"
    docker push "$DEVELOPMENT_LATEST"
    
    print_success "All images pushed successfully!"
}

# Show image information
show_image_info() {
    print_info "Built images:"
    echo
    docker images | grep "$DOCKER_USERNAME/$IMAGE_NAME"
    echo
    
    print_info "Image details:"
    echo "Production: $PRODUCTION_IMAGE"
    echo "Production Latest: $PRODUCTION_LATEST"
    echo "Development: $DEVELOPMENT_IMAGE"
    echo "Development Latest: $DEVELOPMENT_LATEST"
    echo
    echo "Build Date: $BUILD_DATE"
    echo "Version: $VERSION"
    echo "VCS Ref: $VCS_REF"
}

# Clean up old images
cleanup_old_images() {
    print_info "Cleaning up old images..."
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old versions (keep last 3)
    docker images "$DOCKER_USERNAME/$IMAGE_NAME" --format "table {{.Tag}}\t{{.ID}}" | \
        grep -E '^[0-9]+\.[0-9]+\.[0-9]+' | \
        tail -n +4 | \
        awk '{print $2}' | \
        xargs -r docker rmi
    
    print_success "Cleanup completed"
}

# Show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -v, --version       Set version (default: $VERSION)"
    echo "  -u, --username      Set Docker Hub username (default: $DOCKER_USERNAME)"
    echo "  --prod-only         Build only production image"
    echo "  --dev-only          Build only development image"
    echo "  --no-push           Build but don't push to Docker Hub"
    echo "  --no-test           Skip image testing"
    echo "  --cleanup           Clean up old images after build"
    echo
    echo "Examples:"
    echo "  $0                                    # Build and push all images"
    echo "  $0 --version 2.0.0                   # Build with specific version"
    echo "  $0 --prod-only --no-push             # Build only production, don't push"
    echo "  $0 --username myusername              # Use different Docker Hub username"
}

# Parse command line arguments
parse_args() {
    BUILD_PROD=true
    BUILD_DEV=true
    PUSH_IMAGES=true
    TEST_IMAGES=true
    CLEANUP_IMAGES=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -u|--username)
                DOCKER_USERNAME="$2"
                shift 2
                ;;
            --prod-only)
                BUILD_DEV=false
                shift
                ;;
            --dev-only)
                BUILD_PROD=false
                shift
                ;;
            --no-push)
                PUSH_IMAGES=false
                shift
                ;;
            --no-test)
                TEST_IMAGES=false
                shift
                ;;
            --cleanup)
                CLEANUP_IMAGES=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Update image names with new username/version
    PRODUCTION_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
    PRODUCTION_LATEST="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
    DEVELOPMENT_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-dev"
    DEVELOPMENT_LATEST="${DOCKER_USERNAME}/${IMAGE_NAME}:dev"
}

# Main execution
main() {
    parse_args "$@"
    
    print_header
    
    print_info "Configuration:"
    echo "Docker Username: $DOCKER_USERNAME"
    echo "Image Name: $IMAGE_NAME"
    echo "Version: $VERSION"
    echo "Build Production: $BUILD_PROD"
    echo "Build Development: $BUILD_DEV"
    echo "Push Images: $PUSH_IMAGES"
    echo "Test Images: $TEST_IMAGES"
    echo "Cleanup: $CLEANUP_IMAGES"
    echo
    
    check_prerequisites
    
    # Build images
    if [ "$BUILD_PROD" = true ]; then
        build_production
    fi
    
    if [ "$BUILD_DEV" = true ]; then
        build_development
    fi
    
    # Test images
    if [ "$TEST_IMAGES" = true ]; then
        test_images
    fi
    
    # Push images
    if [ "$PUSH_IMAGES" = true ]; then
        push_images
    fi
    
    # Show information
    show_image_info
    
    # Cleanup if requested
    if [ "$CLEANUP_IMAGES" = true ]; then
        cleanup_old_images
    fi
    
    print_success "Docker build and push completed successfully! 🎉"
    
    if [ "$PUSH_IMAGES" = true ]; then
        echo
        print_info "Your images are now available on Docker Hub:"
        echo "docker pull $PRODUCTION_LATEST"
        echo "docker pull $DEVELOPMENT_LATEST"
        echo
        print_info "To run the application:"
        echo "docker run -p 8501:8501 $PRODUCTION_LATEST"
    fi
}

# Run main function
main "$@"
