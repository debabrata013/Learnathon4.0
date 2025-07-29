#!/bin/bash

# Auto Insurance Fraud Detection System - Staging Deployment Script
# =================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration from environment variables
DOCKER_USERNAME="${DOCKER_USERNAME:-debabratapattnayak}"
IMAGE_NAME="${IMAGE_NAME:-fraud-detection-system}"
IMAGE_TAG="${CI_COMMIT_SHORT_SHA:-latest}"
STAGING_SERVER="${STAGING_SERVER:-staging.fraud-detection.example.com}"
STAGING_USER="${STAGING_USER:-deploy}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.staging.yml}"

print_header() {
    echo -e "${BLUE}"
    echo "🚀 =============================================="
    echo "   Staging Deployment Script"
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

# Setup SSH key for deployment
setup_ssh() {
    print_info "Setting up SSH connection..."
    
    # Create SSH directory
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    
    # Add SSH key if provided
    if [ -n "$STAGING_SSH_KEY" ]; then
        echo "$STAGING_SSH_KEY" > ~/.ssh/staging_key
        chmod 600 ~/.ssh/staging_key
        
        # Add to SSH config
        cat >> ~/.ssh/config << EOF
Host staging-server
    HostName $STAGING_SERVER
    User $STAGING_USER
    IdentityFile ~/.ssh/staging_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
        
        print_success "SSH key configured"
    else
        print_warning "No SSH key provided, using default authentication"
    fi
}

# Create staging docker-compose file
create_staging_compose() {
    print_info "Creating staging docker-compose configuration..."
    
    cat > docker-compose.staging.yml << EOF
version: '3.8'

services:
  fraud-detection-app:
    image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
    container_name: fraud-detection-staging
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=staging
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - GEMINI_API_KEY=\${GEMINI_API_KEY}
    volumes:
      - ./staging-data:/app/dataset:ro
      - ./staging-outputs:/app/outputs
      - ./staging-logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "/app/healthcheck.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - fraud-detection-staging

  redis-cache:
    image: redis:7-alpine
    container_name: fraud-redis-staging
    ports:
      - "6379:6379"
    volumes:
      - redis-staging-data:/data
    command: redis-server --appendonly yes
    networks:
      - fraud-detection-staging

  postgres-db:
    image: postgres:15-alpine
    container_name: fraud-postgres-staging
    environment:
      - POSTGRES_DB=fraud_detection_staging
      - POSTGRES_USER=\${POSTGRES_USER:-fraud_user}
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres-staging-data:/var/lib/postgresql/data
    networks:
      - fraud-detection-staging

networks:
  fraud-detection-staging:
    driver: bridge
    name: fraud-detection-staging

volumes:
  redis-staging-data:
    driver: local
  postgres-staging-data:
    driver: local
EOF

    print_success "Staging docker-compose file created"
}

# Deploy to staging server
deploy_to_staging() {
    print_info "Deploying to staging server..."
    
    # Copy deployment files to staging server
    print_info "Copying deployment files..."
    scp docker-compose.staging.yml staging-server:~/
    
    # Create deployment script on staging server
    cat > deploy-commands.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting staging deployment..."

# Create necessary directories
mkdir -p staging-data staging-outputs staging-logs

# Pull latest image
echo "📥 Pulling latest Docker image..."
docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.staging.yml down || true

# Start new containers
echo "🚀 Starting new containers..."
docker-compose -f docker-compose.staging.yml up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🏥 Performing health check..."
curl -f http://localhost:8501/_stcore/health || exit 1

echo "✅ Staging deployment completed successfully!"
EOF

    # Copy and execute deployment script
    scp deploy-commands.sh staging-server:~/
    ssh staging-server "chmod +x deploy-commands.sh && ./deploy-commands.sh"
    
    print_success "Deployment to staging completed"
}

# Verify deployment
verify_deployment() {
    print_info "Verifying staging deployment..."
    
    # Test health endpoint
    if curl -f "http://${STAGING_SERVER}:8501/_stcore/health" &>/dev/null; then
        print_success "Health check passed"
    else
        print_error "Health check failed"
        return 1
    fi
    
    # Test main application
    if curl -f "http://${STAGING_SERVER}:8501/" &>/dev/null; then
        print_success "Application is responding"
    else
        print_error "Application is not responding"
        return 1
    fi
    
    print_success "Staging deployment verification completed"
}

# Send notification
send_notification() {
    local status=$1
    local message="Staging deployment ${status} for commit ${CI_COMMIT_SHORT_SHA}"
    
    print_info "Sending deployment notification..."
    
    # Slack notification
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🚀 ${message}\"}" \
            "$SLACK_WEBHOOK_URL" || true
    fi
    
    # Discord notification
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"content\":\"🚀 ${message}\"}" \
            "$DISCORD_WEBHOOK_URL" || true
    fi
    
    print_success "Notification sent"
}

# Rollback function
rollback_deployment() {
    print_warning "Rolling back staging deployment..."
    
    ssh staging-server << 'EOF'
# Stop current containers
docker-compose -f docker-compose.staging.yml down

# Start previous version (if available)
if [ -f docker-compose.staging.yml.backup ]; then
    mv docker-compose.staging.yml.backup docker-compose.staging.yml
    docker-compose -f docker-compose.staging.yml up -d
    echo "✅ Rollback completed"
else
    echo "❌ No backup found for rollback"
    exit 1
fi
EOF

    print_success "Rollback completed"
}

# Main execution
main() {
    print_header
    
    print_info "Deployment Configuration:"
    echo "Docker Username: $DOCKER_USERNAME"
    echo "Image Name: $IMAGE_NAME"
    echo "Image Tag: $IMAGE_TAG"
    echo "Staging Server: $STAGING_SERVER"
    echo "Staging User: $STAGING_USER"
    echo
    
    # Trap for cleanup on failure
    trap 'print_error "Deployment failed"; send_notification "FAILED"; exit 1' ERR
    
    # Execute deployment steps
    setup_ssh
    create_staging_compose
    deploy_to_staging
    verify_deployment
    send_notification "SUCCESSFUL"
    
    print_success "🎉 Staging deployment completed successfully!"
    
    echo
    print_info "Staging Environment Access:"
    echo "Dashboard: http://${STAGING_SERVER}:8501"
    echo "Health Check: http://${STAGING_SERVER}:8501/_stcore/health"
}

# Handle rollback if requested
if [ "$1" = "rollback" ]; then
    print_header
    rollback_deployment
    exit 0
fi

# Run main deployment
main "$@"
