#!/bin/bash

# Auto Insurance Fraud Detection System - Production Deployment Script
# ====================================================================

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
IMAGE_TAG="${CI_COMMIT_TAG#v}"
PRODUCTION_SERVER="${PRODUCTION_SERVER:-fraud-detection.example.com}"
PRODUCTION_USER="${PRODUCTION_USER:-deploy}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"

print_header() {
    echo -e "${BLUE}"
    echo "🏭 =============================================="
    echo "   Production Deployment Script"
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

# Pre-deployment checks
pre_deployment_checks() {
    print_info "Running pre-deployment checks..."
    
    # Check if image exists
    if ! docker manifest inspect "${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}" &>/dev/null; then
        print_error "Docker image ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG} not found"
        exit 1
    fi
    
    # Check if production server is accessible
    if ! ping -c 1 "$PRODUCTION_SERVER" &>/dev/null; then
        print_error "Production server $PRODUCTION_SERVER is not accessible"
        exit 1
    fi
    
    # Verify SSH connection
    if ! ssh -o ConnectTimeout=10 production-server "echo 'SSH connection successful'" &>/dev/null; then
        print_error "SSH connection to production server failed"
        exit 1
    fi
    
    print_success "Pre-deployment checks passed"
}

# Setup SSH key for deployment
setup_ssh() {
    print_info "Setting up SSH connection..."
    
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    
    if [ -n "$PRODUCTION_SSH_KEY" ]; then
        echo "$PRODUCTION_SSH_KEY" > ~/.ssh/production_key
        chmod 600 ~/.ssh/production_key
        
        cat >> ~/.ssh/config << EOF
Host production-server
    HostName $PRODUCTION_SERVER
    User $PRODUCTION_USER
    IdentityFile ~/.ssh/production_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
        
        print_success "SSH key configured"
    else
        print_error "Production SSH key not provided"
        exit 1
    fi
}

# Create production docker-compose file
create_production_compose() {
    print_info "Creating production docker-compose configuration..."
    
    cat > docker-compose.production.yml << EOF
version: '3.8'

services:
  fraud-detection-app:
    image: ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}
    container_name: fraud-detection-production
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - GEMINI_API_KEY=\${GEMINI_API_KEY}
      - SENTRY_DSN=\${SENTRY_DSN}
    volumes:
      - ./production-data:/app/dataset:ro
      - ./production-outputs:/app/outputs
      - ./production-logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "/app/healthcheck.sh"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    networks:
      - fraud-detection-production
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  redis-cache:
    image: redis:7-alpine
    container_name: fraud-redis-production
    ports:
      - "6379:6379"
    volumes:
      - redis-production-data:/data
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    restart: unless-stopped
    networks:
      - fraud-detection-production
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  postgres-db:
    image: postgres:15-alpine
    container_name: fraud-postgres-production
    environment:
      - POSTGRES_DB=fraud_detection_production
      - POSTGRES_USER=\${POSTGRES_USER}
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres-production-data:/var/lib/postgresql/data
      - ./backups:/backups
    restart: unless-stopped
    networks:
      - fraud-detection-production
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  nginx-proxy:
    image: nginx:alpine
    container_name: fraud-nginx-production
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - fraud-detection-app
    restart: unless-stopped
    networks:
      - fraud-detection-production
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

networks:
  fraud-detection-production:
    driver: bridge
    name: fraud-detection-production

volumes:
  redis-production-data:
    driver: local
  postgres-production-data:
    driver: local
EOF

    print_success "Production docker-compose file created"
}

# Create backup before deployment
create_backup() {
    print_info "Creating backup before deployment..."
    
    local backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_dir="backup_${backup_timestamp}"
    
    ssh production-server << EOF
# Create backup directory
mkdir -p ~/backups/${backup_dir}

# Backup database
if docker ps | grep -q fraud-postgres-production; then
    echo "📦 Backing up database..."
    docker exec fraud-postgres-production pg_dump -U \$POSTGRES_USER fraud_detection_production > ~/backups/${backup_dir}/database_backup.sql
fi

# Backup application data
if [ -d ~/production-outputs ]; then
    echo "📦 Backing up application data..."
    tar -czf ~/backups/${backup_dir}/application_data.tar.gz production-outputs/ production-logs/ || true
fi

# Backup current docker-compose
if [ -f ~/docker-compose.production.yml ]; then
    echo "📦 Backing up docker-compose configuration..."
    cp ~/docker-compose.production.yml ~/backups/${backup_dir}/
fi

# Clean old backups (keep last 7 days)
find ~/backups -type d -name "backup_*" -mtime +${BACKUP_RETENTION_DAYS} -exec rm -rf {} + || true

echo "✅ Backup created: ${backup_dir}"
EOF

    print_success "Backup created successfully"
}

# Blue-green deployment
blue_green_deploy() {
    print_info "Starting blue-green deployment..."
    
    # Copy deployment files
    scp docker-compose.production.yml production-server:~/
    scp -r nginx/ production-server:~/ || true
    
    # Execute blue-green deployment on production server
    ssh production-server << 'EOF'
#!/bin/bash
set -e

echo "🔵 Starting blue-green deployment..."

# Pull new image
echo "📥 Pulling new Docker image..."
docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}

# Check if green environment exists
if docker ps | grep -q fraud-detection-green; then
    echo "🟢 Green environment exists, switching to blue..."
    NEW_ENV="blue"
    OLD_ENV="green"
else
    echo "🔵 Blue environment exists or first deployment, switching to green..."
    NEW_ENV="green"
    OLD_ENV="blue"
fi

# Start new environment
echo "🚀 Starting ${NEW_ENV} environment..."
sed "s/fraud-detection-production/fraud-detection-${NEW_ENV}/g" docker-compose.production.yml > docker-compose.${NEW_ENV}.yml
sed -i "s/8501:8501/8502:8501/g" docker-compose.${NEW_ENV}.yml

docker-compose -f docker-compose.${NEW_ENV}.yml up -d fraud-detection-app

# Wait for new environment to be ready
echo "⏳ Waiting for ${NEW_ENV} environment to be ready..."
sleep 45

# Health check on new environment
if curl -f http://localhost:8502/_stcore/health; then
    echo "✅ ${NEW_ENV} environment is healthy"
    
    # Switch traffic (update nginx or load balancer)
    echo "🔄 Switching traffic to ${NEW_ENV} environment..."
    
    # Update nginx configuration to point to new environment
    if [ -f nginx/nginx.conf ]; then
        sed -i "s/fraud-detection-production/fraud-detection-${NEW_ENV}/g" nginx/nginx.conf
        docker exec fraud-nginx-production nginx -s reload || true
    fi
    
    # Stop old environment
    if docker ps | grep -q "fraud-detection-${OLD_ENV}"; then
        echo "🛑 Stopping ${OLD_ENV} environment..."
        docker-compose -f docker-compose.${OLD_ENV}.yml down || true
    fi
    
    # Rename new environment to production
    docker rename fraud-detection-${NEW_ENV} fraud-detection-production
    
    echo "✅ Blue-green deployment completed successfully!"
else
    echo "❌ ${NEW_ENV} environment health check failed"
    docker-compose -f docker-compose.${NEW_ENV}.yml down
    exit 1
fi
EOF

    print_success "Blue-green deployment completed"
}

# Verify production deployment
verify_production_deployment() {
    print_info "Verifying production deployment..."
    
    local max_retries=5
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -f "https://${PRODUCTION_SERVER}/_stcore/health" &>/dev/null; then
            print_success "Production health check passed"
            break
        elif curl -f "http://${PRODUCTION_SERVER}:8501/_stcore/health" &>/dev/null; then
            print_success "Production health check passed (HTTP)"
            break
        else
            retry_count=$((retry_count + 1))
            print_warning "Health check failed, retry $retry_count/$max_retries"
            sleep 10
        fi
    done
    
    if [ $retry_count -eq $max_retries ]; then
        print_error "Production health check failed after $max_retries attempts"
        return 1
    fi
    
    # Performance test
    print_info "Running production performance test..."
    if command -v ab &>/dev/null; then
        ab -n 50 -c 5 "https://${PRODUCTION_SERVER}/" > production-performance.txt || true
        print_success "Performance test completed"
    fi
    
    print_success "Production deployment verification completed"
}

# Send deployment notification
send_deployment_notification() {
    local status=$1
    local version=$2
    
    local message="🏭 Production deployment ${status} - Version ${version}"
    
    print_info "Sending deployment notification..."
    
    # Slack notification
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"text\":\"${message}\",
                \"attachments\":[{
                    \"color\":\"$([ "$status" = "SUCCESSFUL" ] && echo "good" || echo "danger")\",
                    \"fields\":[
                        {\"title\":\"Version\",\"value\":\"${version}\",\"short\":true},
                        {\"title\":\"Environment\",\"value\":\"Production\",\"short\":true},
                        {\"title\":\"URL\",\"value\":\"https://${PRODUCTION_SERVER}\",\"short\":false}
                    ]
                }]
            }" \
            "$SLACK_WEBHOOK_URL" || true
    fi
    
    # Email notification
    if [ -n "$TEAM_EMAIL" ] && command -v mail &>/dev/null; then
        echo "Production deployment ${status} for version ${version}" | \
            mail -s "🏭 Production Deployment ${status}" "$TEAM_EMAIL" || true
    fi
    
    print_success "Deployment notification sent"
}

# Rollback function
rollback_production() {
    print_warning "Rolling back production deployment..."
    
    ssh production-server << 'EOF'
# Find latest backup
LATEST_BACKUP=$(ls -t ~/backups/ | head -1)

if [ -n "$LATEST_BACKUP" ]; then
    echo "🔄 Rolling back to backup: $LATEST_BACKUP"
    
    # Stop current containers
    docker-compose -f docker-compose.production.yml down || true
    
    # Restore docker-compose configuration
    if [ -f ~/backups/$LATEST_BACKUP/docker-compose.production.yml ]; then
        cp ~/backups/$LATEST_BACKUP/docker-compose.production.yml ~/
        docker-compose -f docker-compose.production.yml up -d
        echo "✅ Rollback completed"
    else
        echo "❌ No backup configuration found"
        exit 1
    fi
else
    echo "❌ No backups found for rollback"
    exit 1
fi
EOF

    print_success "Production rollback completed"
}

# Main execution
main() {
    print_header
    
    print_info "Production Deployment Configuration:"
    echo "Docker Username: $DOCKER_USERNAME"
    echo "Image Name: $IMAGE_NAME"
    echo "Image Tag: $IMAGE_TAG"
    echo "Production Server: $PRODUCTION_SERVER"
    echo "Production User: $PRODUCTION_USER"
    echo
    
    # Confirmation prompt
    echo -e "${YELLOW}⚠️  This will deploy to PRODUCTION environment!${NC}"
    read -p "Are you sure you want to continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_info "Deployment cancelled by user"
        exit 0
    fi
    
    # Trap for cleanup on failure
    trap 'print_error "Production deployment failed"; send_deployment_notification "FAILED" "$IMAGE_TAG"; exit 1' ERR
    
    # Execute deployment steps
    setup_ssh
    pre_deployment_checks
    create_production_compose
    create_backup
    blue_green_deploy
    verify_production_deployment
    send_deployment_notification "SUCCESSFUL" "$IMAGE_TAG"
    
    print_success "🎉 Production deployment completed successfully!"
    
    echo
    print_info "Production Environment Access:"
    echo "Dashboard: https://${PRODUCTION_SERVER}"
    echo "Health Check: https://${PRODUCTION_SERVER}/_stcore/health"
}

# Handle rollback if requested
if [ "$1" = "rollback" ]; then
    print_header
    rollback_production
    send_deployment_notification "ROLLED_BACK" "previous"
    exit 0
fi

# Run main deployment
main "$@"
