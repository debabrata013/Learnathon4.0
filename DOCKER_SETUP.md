# 🐳 Docker Setup Guide - Auto Insurance Fraud Detection System

## Overview

This Docker setup provides a complete containerized environment for the Auto Insurance Fraud Detection System with multiple services and deployment options.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Streamlit     │  │  Data Processor │  │ Model Trainer│ │
│  │   Dashboard     │  │    Service      │  │   Service    │ │
│  │   (Port 8501)   │  │                 │  │              │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Jupyter Lab   │  │   Redis Cache   │  │ PostgreSQL   │ │
│  │   (Port 8888)   │  │   (Port 6379)   │  │ (Port 5432)  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Nginx Reverse Proxy                        │ │
│  │                (Port 80/443)                            │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available
- 10GB free disk space

## 🚀 Quick Start

### 1. Basic Setup (Streamlit Dashboard Only)

```bash
# Clone and navigate to project
cd /Users/debabratapattnayak/web-dev/learnathon

# Build and run the main application
docker-compose up -d fraud-detection-app

# Access the dashboard
open http://localhost:8501
```

### 2. Complete Development Environment

```bash
# Run all development services
docker-compose --profile development up -d

# Access services:
# - Streamlit Dashboard: http://localhost:8501
# - Jupyter Lab: http://localhost:8888 (token: fraud-detection-2024)
```

### 3. Full Production Stack

```bash
# Run production environment with all services
docker-compose --profile production --profile cache --profile database up -d

# Access via Nginx proxy: http://localhost
```

## 🔧 Service Profiles

### Core Services (Default)
- `fraud-detection-app`: Main Streamlit dashboard

### Development Profile
```bash
docker-compose --profile development up -d
```
- `jupyter-lab`: Jupyter development environment

### Preprocessing Profile
```bash
docker-compose --profile preprocessing up -d
```
- `data-processor`: Data preprocessing service

### Training Profile
```bash
docker-compose --profile training up -d
```
- `model-trainer`: Model training service

### Cache Profile
```bash
docker-compose --profile cache up -d
```
- `redis-cache`: Redis caching service

### Database Profile
```bash
docker-compose --profile database up -d
```
- `postgres-db`: PostgreSQL database

### Production Profile
```bash
docker-compose --profile production up -d
```
- `nginx-proxy`: Nginx reverse proxy with SSL support

## 📁 Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./dataset` | `/app/dataset` | Training data (read-only) |
| `./ml_analysis_reports` | `/app/ml_analysis_reports` | Preprocessing reports |
| `./final-model` | `/app/final-model` | Trained models |
| `./outputs` | `/app/outputs` | Generated outputs |
| `./logs` | `/app/logs` | Application logs |

## 🌐 Port Mapping

| Service | Host Port | Container Port | Purpose |
|---------|-----------|----------------|---------|
| Streamlit | 8501 | 8501 | Main dashboard |
| Jupyter | 8888 | 8888 | Development environment |
| Nginx | 80 | 80 | HTTP proxy |
| Nginx SSL | 443 | 443 | HTTPS proxy |
| PostgreSQL | 5432 | 5432 | Database |
| Redis | 6379 | 6379 | Cache |

## 🔐 Environment Configuration

### 1. Copy Environment Template
```bash
cp .env.example .env
```

### 2. Configure Required Variables
```bash
# Edit .env file
nano .env

# Required settings:
GEMINI_API_KEY=your_actual_api_key
POSTGRES_PASSWORD=your_secure_password
JUPYTER_TOKEN=your_secure_token
```

## 🛠️ Common Commands

### Build Services
```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build fraud-detection-app

# Build without cache
docker-compose build --no-cache
```

### Service Management
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart fraud-detection-app

# View logs
docker-compose logs -f fraud-detection-app

# Scale services
docker-compose up -d --scale fraud-detection-app=3
```

### Data Operations
```bash
# Run data preprocessing
docker-compose --profile preprocessing up data-processor

# Run model training
docker-compose --profile training up model-trainer

# Execute one-time preprocessing
docker-compose run --rm data-processor python comprehensive_fraud_preprocessing.py
```

### Development Commands
```bash
# Access container shell
docker-compose exec fraud-detection-app bash

# Run Jupyter notebook
docker-compose --profile development up -d jupyter-lab

# Execute Python script in container
docker-compose exec fraud-detection-app python final-model/final_xgboost_model.py
```

## 🔍 Monitoring and Debugging

### Health Checks
```bash
# Check service health
docker-compose ps

# View detailed service status
docker inspect fraud-detection-dashboard
```

### Log Analysis
```bash
# View all logs
docker-compose logs

# Follow specific service logs
docker-compose logs -f fraud-detection-app

# View last 100 lines
docker-compose logs --tail=100 fraud-detection-app
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats

# View container processes
docker-compose top
```

## 🚀 Production Deployment

### 1. SSL Configuration
```bash
# Create SSL directory
mkdir -p nginx/ssl

# Add your SSL certificates
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem
```

### 2. Production Environment
```bash
# Set production environment
export ENVIRONMENT=production

# Deploy with all production services
docker-compose --profile production --profile cache --profile database up -d
```

### 3. Backup and Restore
```bash
# Backup database
docker-compose exec postgres-db pg_dump -U fraud_user fraud_detection > backup.sql

# Backup volumes
docker run --rm -v learnathon_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>
```

#### Permission Issues
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./outputs ./logs

# Set correct permissions
chmod -R 755 ./outputs ./logs
```

#### Memory Issues
```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory > 8GB

# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

#### Container Won't Start
```bash
# Check logs
docker-compose logs fraud-detection-app

# Rebuild container
docker-compose build --no-cache fraud-detection-app

# Remove and recreate
docker-compose down
docker-compose up -d
```

## 📊 Performance Optimization

### 1. Resource Limits
Add to docker-compose.yml:
```yaml
services:
  fraud-detection-app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

### 2. Caching Strategy
```bash
# Enable Redis caching
docker-compose --profile cache up -d redis-cache

# Configure application to use Redis
# (Update app configuration)
```

### 3. Load Balancing
```bash
# Scale Streamlit instances
docker-compose up -d --scale fraud-detection-app=3

# Configure Nginx load balancing
# (Update nginx.conf)
```

## 🔒 Security Best Practices

1. **Environment Variables**: Never commit `.env` files
2. **API Keys**: Use Docker secrets for sensitive data
3. **Network Security**: Use custom networks
4. **SSL/TLS**: Enable HTTPS in production
5. **User Permissions**: Run containers as non-root user
6. **Regular Updates**: Keep base images updated

## 📝 Maintenance

### Regular Tasks
```bash
# Update images
docker-compose pull

# Clean up unused resources
docker system prune -a

# Update application
git pull
docker-compose build
docker-compose up -d
```

### Backup Strategy
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec postgres-db pg_dump -U fraud_user fraud_detection > "backup_${DATE}.sql"
tar -czf "volumes_backup_${DATE}.tar.gz" outputs/ logs/ ml_analysis_reports/
```

## 📞 Support

For issues and questions:
1. Check logs: `docker-compose logs`
2. Review this documentation
3. Check Docker and Docker Compose versions
4. Ensure sufficient system resources

## 🎯 Next Steps

1. **Customize Configuration**: Modify `.env` for your environment
2. **Add Monitoring**: Integrate Prometheus/Grafana
3. **CI/CD Pipeline**: Set up automated deployments
4. **Scaling**: Configure horizontal scaling
5. **Security**: Implement additional security measures
