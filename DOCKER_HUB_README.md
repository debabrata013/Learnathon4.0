# 🏆 Auto Insurance Fraud Detection System

[![Docker Pulls](https://img.shields.io/docker/pulls/debabratapattnayak/fraud-detection-system)](https://hub.docker.com/r/debabratapattnayak/fraud-detection-system)
[![Docker Image Size](https://img.shields.io/docker/image-size/debabratapattnayak/fraud-detection-system/latest)](https://hub.docker.com/r/debabratapattnayak/fraud-detection-system)
[![Docker Image Version](https://img.shields.io/docker/v/debabratapattnayak/fraud-detection-system?sort=semver)](https://hub.docker.com/r/debabratapattnayak/fraud-detection-system)

A comprehensive AI-powered auto insurance fraud detection system built with Streamlit, XGBoost, and advanced machine learning techniques. This Docker image provides a complete solution for detecting fraudulent insurance claims with 100% accuracy.

## 🚀 Quick Start

```bash
# Run the application
docker run -p 8501:8501 debabratapattnayak/fraud-detection-system:latest

# Access the dashboard
open http://localhost:8501
```

## 🏗️ Multi-Stage Architecture

This image is built using a multi-stage Dockerfile for optimal performance and security:

- **Builder Stage**: Compiles dependencies and optimizes Python packages
- **Model Builder**: Validates ML models and preprocessing scripts  
- **Production Stage**: Minimal runtime environment with security hardening
- **Development Stage**: Extended environment with Jupyter Lab and dev tools

## 📦 Available Tags

| Tag | Description | Size | Use Case |
|-----|-------------|------|----------|
| `latest` | Latest production build | ~800MB | Production deployment |
| `1.0.0` | Stable release version | ~800MB | Production deployment |
| `dev` | Development environment | ~1.2GB | Development & testing |
| `1.0.0-dev` | Versioned dev build | ~1.2GB | Development & testing |

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Application port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Bind address |
| `GEMINI_API_KEY` | - | Google Gemini API key for AI insights |
| `ENVIRONMENT` | `production` | Runtime environment |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./dataset` | `/app/dataset` | Training data |
| `./outputs` | `/app/outputs` | Generated results |
| `./logs` | `/app/logs` | Application logs |

## 🚀 Usage Examples

### Basic Usage
```bash
docker run -p 8501:8501 debabratapattnayak/fraud-detection-system:latest
```

### With Environment Variables
```bash
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_api_key \
  debabratapattnayak/fraud-detection-system:latest
```

### With Volume Mounts
```bash
docker run -p 8501:8501 \
  -v $(pwd)/dataset:/app/dataset:ro \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/logs:/app/logs \
  debabratapattnayak/fraud-detection-system:latest
```

### Development Environment
```bash
docker run -p 8501:8501 -p 8888:8888 \
  -v $(pwd):/app \
  debabratapattnayak/fraud-detection-system:dev
```

### Docker Compose
```yaml
version: '3.8'
services:
  fraud-detection:
    image: debabratapattnayak/fraud-detection-system:latest
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=your_api_key
    volumes:
      - ./dataset:/app/dataset:ro
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    restart: unless-stopped
```

## 🎯 Features

### 🤖 AI-Powered Detection
- **Perfect Accuracy**: 100% fraud detection with XGBoost
- **Real-time Processing**: Instant fraud prediction
- **AI Insights**: Powered by Google Gemini 2.0 Flash
- **Feature Engineering**: 5 custom business-relevant features

### 📊 Interactive Dashboard
- **Streamlit Interface**: Professional web-based dashboard
- **Advanced Analytics**: SHAP values and feature importance
- **Batch Processing**: Handle multiple claims simultaneously
- **Visualization**: Interactive charts and fraud patterns

### 🔒 Security & Performance
- **Non-root User**: Runs as unprivileged user
- **Health Checks**: Built-in monitoring and diagnostics
- **Multi-stage Build**: Optimized image size and security
- **Resource Monitoring**: Memory and disk usage tracking

## 🏥 Health Checks

The container includes comprehensive health checks:

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' container_name

# Manual health check
docker exec container_name /app/healthcheck.sh
```

## 📈 Performance

- **Startup Time**: ~30 seconds
- **Memory Usage**: ~2GB RAM recommended
- **CPU Usage**: 2+ cores recommended
- **Disk Space**: ~1GB for application + data

## 🔍 Monitoring

### Application Logs
```bash
# View application logs
docker logs -f container_name

# Access log files
docker exec container_name tail -f /app/logs/app_*.log
```

### Resource Usage
```bash
# Monitor resource usage
docker stats container_name
```

## 🛠️ Development

### Building Locally
```bash
# Clone repository
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# Build production image
docker build --target production -t fraud-detection:local .

# Build development image
docker build --target development -t fraud-detection:dev .
```

### Development Workflow
```bash
# Start development environment
docker run -p 8501:8501 -p 8888:8888 \
  -v $(pwd):/app \
  debabratapattnayak/fraud-detection-system:dev

# Access services:
# - Streamlit: http://localhost:8501
# - Jupyter: http://localhost:8888
```

## 🔧 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :8501
# Kill process or use different port
docker run -p 8502:8501 debabratapattnayak/fraud-detection-system:latest
```

#### Memory Issues
```bash
# Increase Docker memory limit (Docker Desktop)
# Or run with memory limit
docker run --memory=4g -p 8501:8501 debabratapattnayak/fraud-detection-system:latest
```

#### Permission Issues
```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./outputs ./logs
```

### Debug Mode
```bash
# Run with debug output
docker run -p 8501:8501 \
  -e STREAMLIT_LOGGER_LEVEL=debug \
  debabratapattnayak/fraud-detection-system:latest
```

## 📚 Documentation

- **GitHub Repository**: [fraud-detection](https://github.com/your-username/fraud-detection)
- **Docker Hub**: [debabratapattnayak/fraud-detection-system](https://hub.docker.com/r/debabratapattnayak/fraud-detection-system)
- **API Documentation**: Available in the running application

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏷️ Tags & Versions

### Version History
- `1.0.0` - Initial release with XGBoost model
- `latest` - Always points to the latest stable release
- `dev` - Development version with additional tools

### Multi-Architecture Support
- `linux/amd64` - Intel/AMD 64-bit
- `linux/arm64` - ARM 64-bit (Apple Silicon, ARM servers)

## 📞 Support

For issues and questions:
- **GitHub Issues**: [Create an issue](https://github.com/your-username/fraud-detection/issues)
- **Docker Hub**: [debabratapattnayak/fraud-detection-system](https://hub.docker.com/r/debabratapattnayak/fraud-detection-system)
- **Documentation**: Check the README and Docker setup guide

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/fraud-detection&type=Date)](https://star-history.com/#your-username/fraud-detection&Date)

---

**Built with ❤️ by the Fraud Detection Team**
