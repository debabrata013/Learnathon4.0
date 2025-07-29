# 🦊 GitLab CI/CD Setup Guide - Auto Insurance Fraud Detection System

## 📋 Overview

This guide provides comprehensive instructions for setting up GitLab CI/CD pipeline for the Auto Insurance Fraud Detection System with multi-stage Docker builds, security scanning, and automated deployments.

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitLab CI/CD Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│  🔍 Validate → 🏗️ Build → 🧪 Test → 🔒 Security → 📦 Package  │
│                           ↓                                     │
│              🚀 Deploy Staging → 🚀 Deploy Production          │
│                           ↓                                     │
│                    🧹 Cleanup → 📊 Reports                     │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Setup

### 1. Repository Setup
```bash
# Add GitLab CI files to your repository
git add .gitlab-ci.yml
git add .gitlab-ci-variables.yml
git add scripts/deploy-*.sh
git commit -m "Add GitLab CI/CD pipeline"
git push origin main
```

### 2. Configure Variables
Go to **Project Settings > CI/CD > Variables** and add:

#### 🔒 Required Variables
| Variable | Value | Protected | Masked |
|----------|-------|-----------|--------|
| `DOCKER_USERNAME` | `debabratapattnayak` | ❌ | ❌ |
| `DOCKER_PASSWORD` | Your Docker Hub token | ✅ | ✅ |
| `GEMINI_API_KEY` | Your Gemini API key | ✅ | ✅ |

#### 🌐 Deployment Variables (Optional)
| Variable | Value | Protected | Masked |
|----------|-------|-----------|--------|
| `STAGING_SERVER` | `staging.example.com` | ❌ | ❌ |
| `PRODUCTION_SERVER` | `production.example.com` | ✅ | ❌ |
| `STAGING_SSH_KEY` | SSH private key | ✅ | ✅ |
| `PRODUCTION_SSH_KEY` | SSH private key | ✅ | ✅ |

### 3. Enable GitLab Runner
Ensure your project has access to GitLab Runners with Docker support.

## 📊 Pipeline Stages

### 🔍 Stage 1: Validation
- **Dockerfile Linting**: Hadolint validation
- **Docker Compose Validation**: Syntax checking
- **Python Syntax**: Code validation

### 🏗️ Stage 2: Build
- **Multi-stage Docker Build**: Production and development images
- **Multi-architecture**: AMD64 and ARM64 support
- **Layer Caching**: Optimized build times

### 🧪 Stage 3: Testing
- **Functionality Tests**: Health checks and API testing
- **Performance Tests**: Load testing with Apache Bench
- **Integration Tests**: End-to-end testing

### 🔒 Stage 4: Security
- **Trivy Scanning**: Vulnerability assessment
- **Secret Scanning**: TruffleHog integration
- **Container Security**: Best practices validation

### 📦 Stage 5: Packaging
- **Image Tagging**: Version and latest tags
- **Registry Push**: Docker Hub deployment
- **Artifact Management**: Build artifacts storage

### 🚀 Stage 6: Deployment
- **Staging Deployment**: Automated staging deployment
- **Production Deployment**: Manual production deployment
- **Blue-Green Deployment**: Zero-downtime deployments

## 🔧 Pipeline Configuration

### Branch Strategy
```yaml
# Main branch (production)
main:
  - Full pipeline execution
  - Automatic staging deployment
  - Manual production deployment

# Develop branch (development)
develop:
  - Build and test
  - Development image creation
  - Staging deployment

# Feature branches
feature/*:
  - Build and test only
  - No deployment

# Release tags
v*.*.*:
  - Full pipeline
  - Automatic production deployment
```

### Environment Variables
```yaml
# Global variables in .gitlab-ci.yml
variables:
  DOCKER_DRIVER: overlay2
  REGISTRY: docker.io
  IMAGE_NAME: fraud-detection-system
  APP_VERSION: "1.0.0"
```

## 🏃‍♂️ Running the Pipeline

### Automatic Triggers
- **Push to main**: Full pipeline with staging deployment
- **Push to develop**: Build, test, and development deployment
- **Create tag**: Release pipeline with production deployment
- **Merge request**: Build and test only

### Manual Triggers
```bash
# Trigger pipeline manually
curl -X POST \
  -F token=YOUR_TRIGGER_TOKEN \
  -F ref=main \
  https://gitlab.com/api/v4/projects/PROJECT_ID/trigger/pipeline
```

### Pipeline Variables Override
```bash
# Run with custom variables
curl -X POST \
  -F token=YOUR_TRIGGER_TOKEN \
  -F ref=main \
  -F "variables[VERSION]=2.0.0" \
  -F "variables[DEPLOY_PRODUCTION]=true" \
  https://gitlab.com/api/v4/projects/PROJECT_ID/trigger/pipeline
```

## 🔍 Monitoring and Debugging

### Pipeline Status
- **GitLab UI**: Project > CI/CD > Pipelines
- **Job Logs**: Click on individual jobs for detailed logs
- **Artifacts**: Download build artifacts and reports

### Common Issues and Solutions

#### 🐳 Docker Build Failures
```bash
# Check Docker daemon
docker info

# Clear build cache
docker builder prune -a

# Check Dockerfile syntax
hadolint Dockerfile
```

#### 🔑 Authentication Issues
```bash
# Verify Docker Hub credentials
docker login docker.io

# Check GitLab variables
# Project Settings > CI/CD > Variables
```

#### 🌐 Deployment Failures
```bash
# Check SSH connectivity
ssh -T git@gitlab.com

# Verify server access
ping staging.example.com

# Check deployment logs
gitlab-runner logs
```

## 📊 Reports and Artifacts

### Available Reports
- **Security Scan**: Trivy vulnerability report
- **Code Quality**: Hadolint Dockerfile analysis
- **Performance**: Load testing results
- **Test Coverage**: Unit test coverage reports

### Accessing Reports
1. Go to **Project > CI/CD > Pipelines**
2. Click on pipeline number
3. Navigate to **Tests** or **Security** tabs
4. Download artifacts from job pages

## 🔒 Security Best Practices

### Variable Security
- ✅ Mark sensitive variables as **Masked**
- ✅ Mark production variables as **Protected**
- ✅ Use environment-specific variables
- ✅ Regularly rotate secrets

### Image Security
- ✅ Multi-stage builds for minimal attack surface
- ✅ Non-root user in containers
- ✅ Regular vulnerability scanning
- ✅ Signed images (optional)

### Deployment Security
- ✅ SSH key-based authentication
- ✅ Network security groups
- ✅ HTTPS/TLS encryption
- ✅ Backup and rollback procedures

## 🚀 Advanced Features

### Parallel Jobs
```yaml
# Run tests in parallel
test:unit:
  stage: test
  parallel: 3
  script:
    - pytest tests/ --junitxml=report.xml
```

### Conditional Deployments
```yaml
# Deploy only on specific conditions
deploy:production:
  rules:
    - if: $CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+$/
      when: manual
    - if: $CI_COMMIT_BRANCH == "main" && $DEPLOY_PRODUCTION == "true"
```

### Custom Runners
```yaml
# Use specific runner tags
build:custom:
  tags:
    - docker
    - high-memory
  script:
    - docker build .
```

## 📈 Performance Optimization

### Build Optimization
- **Layer Caching**: Use registry cache
- **Multi-stage Builds**: Reduce image size
- **Parallel Builds**: Use buildx for multi-arch

### Pipeline Optimization
- **Job Dependencies**: Minimize unnecessary jobs
- **Artifact Management**: Clean up old artifacts
- **Resource Limits**: Configure appropriate limits

## 🔧 Troubleshooting

### Pipeline Debugging
```bash
# Enable debug mode
export CI_DEBUG_TRACE=true

# Check runner status
gitlab-runner status

# Verify GitLab connectivity
curl -H "PRIVATE-TOKEN: your_token" \
  https://gitlab.com/api/v4/projects/PROJECT_ID
```

### Common Error Solutions

#### "Docker daemon not available"
```yaml
services:
  - docker:24.0.5-dind
variables:
  DOCKER_HOST: tcp://docker:2376
  DOCKER_TLS_VERIFY: 1
```

#### "Permission denied" on scripts
```bash
# Make scripts executable
chmod +x scripts/*.sh
git add scripts/
git commit -m "Fix script permissions"
```

#### "Image not found" during deployment
```bash
# Verify image exists
docker manifest inspect username/image:tag

# Check registry credentials
docker login docker.io
```

## 📞 Support and Resources

### GitLab Documentation
- [GitLab CI/CD](https://docs.gitlab.com/ee/ci/)
- [Docker Integration](https://docs.gitlab.com/ee/ci/docker/)
- [Variables](https://docs.gitlab.com/ee/ci/variables/)

### Project Resources
- **Repository**: Your GitLab repository
- **Docker Hub**: [debabratapattnayak/fraud-detection-system](https://hub.docker.com/r/debabratapattnayak/fraud-detection-system)
- **Issues**: GitLab Issues for bug reports

### Getting Help
1. Check pipeline logs in GitLab UI
2. Review this documentation
3. Check GitLab CI/CD documentation
4. Create issue in project repository

## 🎯 Next Steps

1. **Setup Variables**: Configure all required CI/CD variables
2. **Test Pipeline**: Push code and verify pipeline execution
3. **Configure Deployments**: Set up staging and production servers
4. **Monitor Performance**: Review pipeline execution times
5. **Optimize**: Improve build times and resource usage

---

**Happy CI/CD! 🚀**
