#!/bin/bash

# Auto Insurance Fraud Detection System - Health Check Script
# ===========================================================

set -e

# Configuration
HEALTH_CHECK_URL="http://localhost:${STREAMLIT_SERVER_PORT:-8501}/_stcore/health"
TIMEOUT=10
MAX_RETRIES=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" >&2
}

log_success() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${GREEN}SUCCESS${NC}: $1" >&2
}

log_warning() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${YELLOW}WARNING${NC}: $1" >&2
}

log_error() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] ${RED}ERROR${NC}: $1" >&2
}

# Check if curl is available
check_curl() {
    if ! command -v curl >/dev/null 2>&1; then
        log_error "curl is not available for health check"
        return 1
    fi
    return 0
}

# Check Streamlit health endpoint
check_streamlit_health() {
    local retry_count=0
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        if curl -f -s --max-time $TIMEOUT "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
            log_success "Streamlit health check passed"
            return 0
        fi
        
        retry_count=$((retry_count + 1))
        log_warning "Health check attempt $retry_count failed, retrying..."
        sleep 2
    done
    
    log_error "Streamlit health check failed after $MAX_RETRIES attempts"
    return 1
}

# Check Python process
check_python_process() {
    if pgrep -f "streamlit" >/dev/null 2>&1; then
        log_success "Streamlit process is running"
        return 0
    else
        log_error "Streamlit process not found"
        return 1
    fi
}

# Check memory usage
check_memory_usage() {
    if command -v free >/dev/null 2>&1; then
        local mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
        local mem_usage_int=${mem_usage%.*}
        
        if [ "$mem_usage_int" -gt 90 ]; then
            log_warning "High memory usage: ${mem_usage}%"
            return 1
        else
            log_info "Memory usage: ${mem_usage}%"
            return 0
        fi
    else
        log_info "Memory check skipped (free command not available)"
        return 0
    fi
}

# Check disk space
check_disk_space() {
    local disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ "$disk_usage" -gt 90 ]; then
        log_warning "High disk usage: ${disk_usage}%"
        return 1
    else
        log_info "Disk usage: ${disk_usage}%"
        return 0
    fi
}

# Check application files
check_app_files() {
    local required_files=(
        "/app/streamlit-app/app.py"
        "/app/streamlit-app/requirements.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "Required file missing: $file"
            return 1
        fi
    done
    
    log_success "All required application files present"
    return 0
}

# Check log directory
check_log_directory() {
    if [ ! -d "/app/logs" ]; then
        log_warning "Log directory not found, creating..."
        mkdir -p /app/logs
    fi
    
    if [ ! -w "/app/logs" ]; then
        log_error "Log directory is not writable"
        return 1
    fi
    
    log_success "Log directory is accessible"
    return 0
}

# Main health check function
main_health_check() {
    local exit_code=0
    
    log_info "Starting comprehensive health check..."
    
    # Check curl availability
    if ! check_curl; then
        exit_code=1
    fi
    
    # Check application files
    if ! check_app_files; then
        exit_code=1
    fi
    
    # Check log directory
    if ! check_log_directory; then
        exit_code=1
    fi
    
    # Check Python process
    if ! check_python_process; then
        exit_code=1
    fi
    
    # Check Streamlit health endpoint (only if curl is available)
    if command -v curl >/dev/null 2>&1; then
        if ! check_streamlit_health; then
            exit_code=1
        fi
    fi
    
    # Check system resources
    if ! check_memory_usage; then
        log_warning "Memory usage check failed, but continuing..."
    fi
    
    if ! check_disk_space; then
        log_warning "Disk space check failed, but continuing..."
    fi
    
    # Final status
    if [ $exit_code -eq 0 ]; then
        log_success "All health checks passed ✅"
        echo "healthy"
    else
        log_error "Health check failed ❌"
        echo "unhealthy"
    fi
    
    return $exit_code
}

# Execute main health check
main_health_check
