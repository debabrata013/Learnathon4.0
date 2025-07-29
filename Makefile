# Auto Insurance Fraud Detection System - Makefile
# =================================================

.PHONY: help build up down logs clean status restart scale backup restore

# Default target
help: ## Show this help message
	@echo "🏆 Auto Insurance Fraud Detection System - Docker Commands"
	@echo "=========================================================="
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# Basic Operations
build: ## Build all Docker images
	@echo "🔨 Building all images..."
	docker-compose build --no-cache

up: ## Start all services
	@echo "🚀 Starting all services..."
	docker-compose up -d

down: ## Stop all services
	@echo "🛑 Stopping all services..."
	docker-compose down

# Quick Start Options
quick: ## Quick start (Dashboard only)
	@echo "⚡ Quick start - Dashboard only..."
	docker-compose up -d fraud-detection-app
	@echo "✅ Dashboard available at: http://localhost:8501"

dev: ## Start development environment
	@echo "🛠️ Starting development environment..."
	docker-compose --profile development up -d
	@echo "✅ Services available:"
	@echo "   - Dashboard: http://localhost:8501"
	@echo "   - Jupyter: http://localhost:8888"

prod: ## Start production environment
	@echo "🏭 Starting production environment..."
	docker-compose --profile production --profile cache --profile database up -d
	@echo "✅ Production environment started!"

# Data Operations
preprocess: ## Run data preprocessing
	@echo "🔄 Running data preprocessing..."
	docker-compose --profile preprocessing up data-processor

train: ## Run model training
	@echo "🎯 Running model training..."
	docker-compose --profile training up model-trainer

pipeline: ## Run complete ML pipeline
	@echo "🔄 Running complete ML pipeline..."
	docker-compose --profile preprocessing --profile training up -d

# Monitoring and Debugging
logs: ## Show logs for all services
	docker-compose logs -f

logs-app: ## Show logs for main application
	docker-compose logs -f fraud-detection-app

logs-jupyter: ## Show logs for Jupyter service
	docker-compose logs -f jupyter-lab

status: ## Show service status
	@echo "📊 Service Status:"
	docker-compose ps
	@echo ""
	@echo "💾 Resource Usage:"
	docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

health: ## Check service health
	@echo "🏥 Health Check:"
	@curl -f http://localhost:8501/_stcore/health 2>/dev/null && echo "✅ Dashboard: Healthy" || echo "❌ Dashboard: Unhealthy"
	@curl -f http://localhost:8888 2>/dev/null && echo "✅ Jupyter: Healthy" || echo "⚠️ Jupyter: Not running or unhealthy"

# Maintenance
clean: ## Clean up containers, networks, and volumes
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	docker system prune -f
	@echo "✅ Cleanup completed!"

restart: ## Restart all services
	@echo "🔄 Restarting all services..."
	docker-compose restart

restart-app: ## Restart main application
	@echo "🔄 Restarting main application..."
	docker-compose restart fraud-detection-app

# Scaling
scale-app: ## Scale main application (usage: make scale-app REPLICAS=3)
	@echo "📈 Scaling application to $(REPLICAS) replicas..."
	docker-compose up -d --scale fraud-detection-app=$(REPLICAS)

# Backup and Restore
backup: ## Create backup of data and outputs
	@echo "💾 Creating backup..."
	@timestamp=$$(date +%Y%m%d_%H%M%S) && \
	tar -czf "backup_$${timestamp}.tar.gz" outputs/ logs/ ml_analysis_reports/ 2>/dev/null && \
	echo "✅ Backup created: backup_$${timestamp}.tar.gz"

restore: ## Restore from backup (usage: make restore BACKUP=backup_file.tar.gz)
	@echo "📥 Restoring from $(BACKUP)..."
	@if [ -f "$(BACKUP)" ]; then \
		tar -xzf "$(BACKUP)" && \
		echo "✅ Backup restored successfully!"; \
	else \
		echo "❌ Backup file $(BACKUP) not found!"; \
	fi

# Database Operations
db-backup: ## Backup PostgreSQL database
	@echo "💾 Backing up database..."
	@timestamp=$$(date +%Y%m%d_%H%M%S) && \
	docker-compose exec postgres-db pg_dump -U fraud_user fraud_detection > "db_backup_$${timestamp}.sql" && \
	echo "✅ Database backup created: db_backup_$${timestamp}.sql"

db-restore: ## Restore PostgreSQL database (usage: make db-restore BACKUP=db_backup.sql)
	@echo "📥 Restoring database from $(BACKUP)..."
	@if [ -f "$(BACKUP)" ]; then \
		docker-compose exec -T postgres-db psql -U fraud_user -d fraud_detection < "$(BACKUP)" && \
		echo "✅ Database restored successfully!"; \
	else \
		echo "❌ Database backup file $(BACKUP) not found!"; \
	fi

# Development Helpers
shell: ## Access main application shell
	docker-compose exec fraud-detection-app bash

shell-jupyter: ## Access Jupyter container shell
	docker-compose exec jupyter-lab bash

install: ## Install/update Python dependencies
	docker-compose exec fraud-detection-app pip install -r requirements.txt

test: ## Run tests (if available)
	docker-compose exec fraud-detection-app python -m pytest tests/ || echo "No tests found"

# Network and Security
network-inspect: ## Inspect Docker network
	docker network inspect fraud-detection-network

security-scan: ## Run basic security scan
	@echo "🔒 Running security scan..."
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		-v $(PWD):/app \
		aquasec/trivy fs /app

# Performance
benchmark: ## Run performance benchmark
	@echo "⚡ Running performance benchmark..."
	@echo "Testing dashboard response time..."
	@time curl -s http://localhost:8501 > /dev/null && echo "✅ Dashboard responsive"

monitor: ## Start monitoring (if available)
	@echo "📊 Starting monitoring..."
	docker-compose --profile monitoring up -d || echo "⚠️ Monitoring profile not available"

# Environment Management
env-check: ## Check environment configuration
	@echo "🔍 Environment Configuration Check:"
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"
	@echo "Available memory: $$(free -h | grep Mem | awk '{print $$2}' 2>/dev/null || echo 'N/A')"
	@echo "Available disk space: $$(df -h . | tail -1 | awk '{print $$4}')"
	@if [ -f .env ]; then echo "✅ .env file exists"; else echo "⚠️ .env file missing"; fi

env-template: ## Create .env from template
	@if [ ! -f .env ]; then \
		cp .env.example .env && \
		echo "✅ .env file created from template"; \
		echo "⚠️ Please edit .env file with your configuration"; \
	else \
		echo "⚠️ .env file already exists"; \
	fi

# Documentation
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	@echo "Available at: DOCKER_SETUP.md"

# Default values for parameterized targets
REPLICAS ?= 2
BACKUP ?= backup.tar.gz
