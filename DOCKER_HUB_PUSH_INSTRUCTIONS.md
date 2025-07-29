# 🐳 Docker Hub Push Instructions

## Step-by-Step Guide to Push Your Fraud Detection System to Docker Hub

### Prerequisites ✅
- Docker Desktop installed and running
- Docker Hub account (username: debabratapattnayak)
- Project files ready

### Step 1: Login to Docker Hub 🔑

Open your terminal and run:
```bash
docker login
```

Enter your credentials:
- **Username**: `debabratapattnayak`
- **Password**: Your Docker Hub password or access token

### Step 2: Navigate to Project Directory 📁
```bash
cd /Users/debabratapattnayak/web-dev/learnathon
```

### Step 3: Run the Quick Push Script 🚀
```bash
./quick-push.sh
```

**OR** follow manual steps below:

### Manual Steps (Alternative) 🛠️

#### Build Production Image:
```bash
docker build \
  --target production \
  --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg VERSION="1.0.0" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --tag debabratapattnayak/fraud-detection-system:1.0.0 \
  --tag debabratapattnayak/fraud-detection-system:latest \
  .
```

#### Build Development Image:
```bash
docker build \
  --target development \
  --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg VERSION="1.0.0" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --tag debabratapattnayak/fraud-detection-system:1.0.0-dev \
  --tag debabratapattnayak/fraud-detection-system:dev \
  .
```

#### Test the Image:
```bash
# Start container
docker run -d --name fraud-test -p 8501:8501 debabratapattnayak/fraud-detection-system:latest

# Wait 30 seconds, then test
sleep 30
curl http://localhost:8501/_stcore/health

# Stop test container
docker stop fraud-test && docker rm fraud-test
```

#### Push to Docker Hub:
```bash
# Push production images
docker push debabratapattnayak/fraud-detection-system:1.0.0
docker push debabratapattnayak/fraud-detection-system:latest

# Push development images
docker push debabratapattnayak/fraud-detection-system:1.0.0-dev
docker push debabratapattnayak/fraud-detection-system:dev
```

### Step 4: Verify on Docker Hub 🔍

1. Go to [Docker Hub](https://hub.docker.com)
2. Login to your account
3. Navigate to your repository: `debabratapattnayak/fraud-detection-system`
4. Verify all tags are present:
   - `latest`
   - `1.0.0`
   - `dev`
   - `1.0.0-dev`

### Step 5: Test Pull and Run 🧪

Test that others can pull and run your image:
```bash
# Pull and run production image
docker pull debabratapattnayak/fraud-detection-system:latest
docker run -p 8501:8501 debabratapattnayak/fraud-detection-system:latest

# Access at: http://localhost:8501
```

### Expected Results 🎯

After successful push, you should see:
- ✅ 4 image tags on Docker Hub
- ✅ Repository publicly accessible
- ✅ Images can be pulled by anyone
- ✅ Application runs successfully

### Troubleshooting 🔧

#### Login Issues:
```bash
# If login fails, try:
docker logout
docker login

# Or use access token instead of password
```

#### Build Issues:
```bash
# If build fails, check:
ls -la Dockerfile
docker system prune -f
```

#### Push Issues:
```bash
# If push fails:
docker images | grep fraud-detection-system
docker tag local-image debabratapattnayak/fraud-detection-system:latest
```

### Docker Hub Repository URL 🌐
After successful push, your repository will be available at:
**https://hub.docker.com/r/debabratapattnayak/fraud-detection-system**

### Usage Examples 📖

Once pushed, anyone can use your image:

```bash
# Basic usage
docker run -p 8501:8501 debabratapattnayak/fraud-detection-system:latest

# With environment variables
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key \
  debabratapattnayak/fraud-detection-system:latest

# Development environment
docker run -p 8501:8501 -p 8888:8888 \
  debabratapattnayak/fraud-detection-system:dev

# Docker Compose
version: '3.8'
services:
  fraud-detection:
    image: debabratapattnayak/fraud-detection-system:latest
    ports:
      - "8501:8501"
    environment:
      - GEMINI_API_KEY=your_key
```

### Next Steps 🚀

1. **Push to Docker Hub** using the steps above
2. **Update README** with Docker Hub links
3. **Test deployment** on different machines
4. **Set up CI/CD** for automated pushes
5. **Monitor usage** via Docker Hub analytics

---

**Ready to push? Run the commands above! 🐳**
