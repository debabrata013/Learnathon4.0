# Multi-stage Dockerfile for Auto Insurance Fraud Detection System
# =================================================================

# Stage 1: Build Dependencies and Compile
FROM python:3.9-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Labels for metadata
LABEL maintainer="Auto Insurance Fraud Detection Team" \
      version="${VERSION}" \
      description="Auto Insurance Fraud Detection System - Builder Stage" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Set environment variables for build
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libgomp1 \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY streamlit-app/requirements.txt /tmp/requirements.txt

# Install Python dependencies in virtual environment
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-compile Python files
COPY streamlit-app/ /tmp/app/
RUN python -m compileall /tmp/app/

# Stage 2: Model Training and Preprocessing (Optional)
FROM python:3.9-slim as model-builder

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy model training scripts
COPY final-model/ /tmp/final-model/
COPY fraud_preprocessing_updated.py /tmp/

# Pre-process and validate model files (if needed)
WORKDIR /tmp
RUN python -c "import xgboost; import joblib; print('Model dependencies verified')"

# Stage 3: Production Runtime
FROM python:3.9-slim as production

# Set build arguments for final stage
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Enhanced labels for production image
LABEL maintainer="Auto Insurance Fraud Detection Team" \
      version="${VERSION}" \
      description="Auto Insurance Fraud Detection System - Production Ready" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}" \
      org.opencontainers.image.title="Fraud Detection Dashboard" \
      org.opencontainers.image.description="AI-powered auto insurance fraud detection system" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Fraud Detection Team" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/your-username/fraud-detection"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true \
    PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/data /app/logs /app/outputs /app/cache && \
    chown -R appuser:appuser /app

# Copy application files with optimized layering
COPY --chown=appuser:appuser streamlit-app/ /app/streamlit-app/
COPY --chown=appuser:appuser final-model/ /app/final-model/
COPY --chown=appuser:appuser dataset/ /app/dataset/
COPY --chown=appuser:appuser ml_analysis_reports/ /app/ml_analysis_reports/

# Copy startup scripts
COPY --chown=appuser:appuser --chmod=755 docker-entrypoint.sh /app/
COPY --chown=appuser:appuser --chmod=755 healthcheck.sh /app/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Add health check with custom script
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Use custom entrypoint for better initialization
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["streamlit", "run", "streamlit-app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 4: Development Environment
FROM production as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development packages
RUN /opt/venv/bin/pip install --no-cache-dir \
    jupyterlab==4.0.9 \
    ipywidgets==8.1.1 \
    pytest==7.4.3 \
    black==23.11.0 \
    flake8==6.1.0 \
    mypy==1.7.1

# Switch back to appuser
USER appuser

# Development environment variables
ENV ENVIRONMENT=development \
    JUPYTER_ENABLE_LAB=yes

# Expose additional ports for development
EXPOSE 8888 8080

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
