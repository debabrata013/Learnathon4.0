#!/bin/bash

# 🏆 Auto Insurance Fraud Detection System - Startup Script
# ========================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Auto Insurance Fraud Detection System"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "🏆 =============================================="
    echo "   $PROJECT_NAME"
    echo "   Docker Deployment Script"
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

check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

setup_environment() {
    print_info "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        print_warning ".env file not found. Creating from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration before proceeding."
        print_info "Required: GEMINI_API_KEY, POSTGRES_PASSWORD, JUPYTER_TOKEN"
        read -p "Press Enter to continue after editing .env file..."
    fi
    
    # Create necessary directories
    mkdir -p outputs logs nginx/ssl
    
    # Set permissions
    chmod +x start.sh
    chmod 755 outputs logs
    
    print_success "Environment setup completed"
}

show_menu() {
    echo
    print_info "Select deployment option:"
    echo "1) 🚀 Quick Start (Streamlit Dashboard only)"
    echo "2) 🛠️  Development Environment (Dashboard + Jupyter)"
    echo "3) 🔄 Data Processing (Preprocessing + Training)"
    echo "4) 🏭 Production Environment (All services)"
    echo "5) 🧹 Clean Up (Stop and remove containers)"
    echo "6) 📊 Show Status"
    echo "7) 📋 Show Logs"
    echo "8) 🔧 Advanced Options"
    echo "9) ❌ Exit"
    echo
}

quick_start() {
    print_info "Starting Quick Start deployment..."
    docker-compose up -d fraud-detection-app
    
    print_success "Quick Start deployment completed!"
    print_info "Access the dashboard at: http://localhost:8501"
}

development_environment() {
    print_info "Starting Development Environment..."
    docker-compose --profile development up -d
    
    print_success "Development Environment started!"
    print_info "Services available:"
    print_info "- Streamlit Dashboard: http://localhost:8501"
    print_info "- Jupyter Lab: http://localhost:8888 (token: fraud-detection-2024)"
}

data_processing() {
    print_info "Starting Data Processing services..."
    docker-compose --profile preprocessing --profile training up -d
    
    print_success "Data Processing services started!"
    print_info "Monitor progress with: docker-compose logs -f data-processor"
}

production_environment() {
    print_info "Starting Production Environment..."
    docker-compose --profile production --profile cache --profile database up -d
    
    print_success "Production Environment started!"
    print_info "Services available:"
    print_info "- Main Application: http://localhost (via Nginx)"
    print_info "- Direct Dashboard: http://localhost:8501"
    print_info "- Database: localhost:5432"
    print_info "- Cache: localhost:6379"
}

clean_up() {
    print_info "Cleaning up containers and volumes..."
    docker-compose down -v
    docker system prune -f
    
    print_success "Clean up completed!"
}

show_status() {
    print_info "Current service status:"
    docker-compose ps
    echo
    print_info "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

show_logs() {
    echo
    print_info "Available services for logs:"
    docker-compose config --services
    echo
    read -p "Enter service name (or 'all' for all services): " service
    
    if [ "$service" = "all" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$service"
    fi
}

advanced_options() {
    echo
    print_info "Advanced Options:"
    echo "1) Build all images"
    echo "2) Pull latest images"
    echo "3) Restart specific service"
    echo "4) Execute command in container"
    echo "5) Backup data"
    echo "6) Restore data"
    echo "7) Scale services"
    echo "8) Back to main menu"
    echo
    
    read -p "Select option (1-8): " adv_choice
    
    case $adv_choice in
        1)
            print_info "Building all images..."
            docker-compose build --no-cache
            print_success "Build completed!"
            ;;
        2)
            print_info "Pulling latest images..."
            docker-compose pull
            print_success "Pull completed!"
            ;;
        3)
            docker-compose config --services
            read -p "Enter service name to restart: " service
            docker-compose restart "$service"
            print_success "Service $service restarted!"
            ;;
        4)
            docker-compose config --services
            read -p "Enter service name: " service
            read -p "Enter command: " command
            docker-compose exec "$service" $command
            ;;
        5)
            print_info "Creating backup..."
            timestamp=$(date +%Y%m%d_%H%M%S)
            tar -czf "backup_${timestamp}.tar.gz" outputs/ logs/ ml_analysis_reports/ 2>/dev/null || true
            print_success "Backup created: backup_${timestamp}.tar.gz"
            ;;
        6)
            ls -la backup_*.tar.gz 2>/dev/null || print_warning "No backup files found"
            read -p "Enter backup file name: " backup_file
            if [ -f "$backup_file" ]; then
                tar -xzf "$backup_file"
                print_success "Backup restored!"
            else
                print_error "Backup file not found!"
            fi
            ;;
        7)
            docker-compose config --services
            read -p "Enter service name: " service
            read -p "Enter number of replicas: " replicas
            docker-compose up -d --scale "$service=$replicas"
            print_success "Service $service scaled to $replicas replicas!"
            ;;
        8)
            return
            ;;
        *)
            print_error "Invalid option!"
            ;;
    esac
}

# Main execution
main() {
    print_header
    check_prerequisites
    setup_environment
    
    while true; do
        show_menu
        read -p "Select option (1-9): " choice
        
        case $choice in
            1)
                quick_start
                ;;
            2)
                development_environment
                ;;
            3)
                data_processing
                ;;
            4)
                production_environment
                ;;
            5)
                clean_up
                ;;
            6)
                show_status
                ;;
            7)
                show_logs
                ;;
            8)
                advanced_options
                ;;
            9)
                print_info "Goodbye! 👋"
                exit 0
                ;;
            *)
                print_error "Invalid option! Please select 1-9."
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"
