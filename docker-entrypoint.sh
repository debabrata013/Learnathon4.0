#!/bin/bash
set -e

# Auto Insurance Fraud Detection System - Docker Entrypoint
# =========================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🏆 Auto Insurance Fraud Detection System${NC}"
echo -e "${BLUE}===========================================${NC}"

# Function to log messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Initialize application
initialize_app() {
    log_info "Initializing application..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/outputs /app/cache
    
    # Set proper permissions
    chmod 755 /app/logs /app/outputs /app/cache
    
    # Check if required files exist
    if [ ! -f "/app/streamlit-app/app.py" ]; then
        log_error "Main application file not found!"
        exit 1
    fi
    
    # Validate Python environment
    python -c "import streamlit, pandas, numpy, plotly, xgboost" || {
        log_error "Required Python packages not available!"
        exit 1
    }
    
    log_success "Application initialized successfully"
}

# Health check function
health_check() {
    log_info "Performing health check..."
    
    # Check Python environment
    python --version
    
    # Check Streamlit installation
    streamlit --version
    
    # Check available memory
    if command -v free >/dev/null 2>&1; then
        free -h
    fi
    
    # Check disk space
    df -h /app
    
    log_success "Health check completed"
}

# Setup logging
setup_logging() {
    log_info "Setting up logging..."
    
    # Create log file with timestamp
    LOG_FILE="/app/logs/app_$(date +%Y%m%d_%H%M%S).log"
    touch "$LOG_FILE"
    
    # Export log file path for application
    export APP_LOG_FILE="$LOG_FILE"
    
    log_success "Logging configured: $LOG_FILE"
}

# Environment validation
validate_environment() {
    log_info "Validating environment..."
    
    # Check required environment variables
    if [ -z "$STREAMLIT_SERVER_PORT" ]; then
        log_warning "STREAMLIT_SERVER_PORT not set, using default: 8501"
        export STREAMLIT_SERVER_PORT=8501
    fi
    
    if [ -z "$STREAMLIT_SERVER_ADDRESS" ]; then
        log_warning "STREAMLIT_SERVER_ADDRESS not set, using default: 0.0.0.0"
        export STREAMLIT_SERVER_ADDRESS=0.0.0.0
    fi
    
    # Validate Gemini API key (if provided)
    if [ -n "$GEMINI_API_KEY" ] && [ "$GEMINI_API_KEY" != "your_gemini_api_key_here" ]; then
        log_success "Gemini API key configured"
    else
        log_warning "Gemini API key not configured - AI insights will be disabled"
    fi
    
    log_success "Environment validation completed"
}

# Pre-flight checks
preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check if port is available
    if command -v netstat >/dev/null 2>&1; then
        if netstat -tuln | grep -q ":$STREAMLIT_SERVER_PORT "; then
            log_warning "Port $STREAMLIT_SERVER_PORT appears to be in use"
        fi
    fi
    
    # Check dataset availability
    if [ -d "/app/dataset" ] && [ "$(ls -A /app/dataset)" ]; then
        log_success "Dataset directory found and not empty"
    else
        log_warning "Dataset directory is empty or missing"
    fi
    
    # Check model files
    if [ -d "/app/final-model" ] && [ "$(ls -A /app/final-model)" ]; then
        log_success "Model files found"
    else
        log_warning "Model files directory is empty or missing"
    fi
    
    log_success "Pre-flight checks completed"
}

# Signal handlers for graceful shutdown
cleanup() {
    log_info "Received shutdown signal, cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    # Save any temporary data
    if [ -f "/tmp/app_state.json" ]; then
        cp /tmp/app_state.json /app/outputs/ 2>/dev/null || true
    fi
    
    log_success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log_info "Starting Auto Insurance Fraud Detection System..."
    
    # Run initialization steps
    initialize_app
    setup_logging
    validate_environment
    preflight_checks
    health_check
    
    log_success "System ready! Starting application..."
    echo -e "${GREEN}🚀 Dashboard will be available at: http://localhost:$STREAMLIT_SERVER_PORT${NC}"
    echo ""
    
    # Execute the main command
    exec "$@"
}

# Run main function with all arguments
main "$@"
