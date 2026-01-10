# 🚀 AWS EC2 Deployment Guide

## Complete Guide: Deploying MSL Recognition API to AWS EC2

This guide walks you through deploying the MSL (Malaysian Sign Language) Recognition API to AWS EC2 with Docker and Nginx reverse proxy.

> **Note:** This guide uses the **API Proxy Pattern** - no SSL required on EC2. The Vercel frontend handles HTTPS.

---

## 📋 Table of Contents

1. [Prerequisites](#-prerequisites)
2. [Architecture Overview](#-architecture-overview)
3. [AWS Setup](#-aws-setup)
4. [EC2 Instance Setup](#-ec2-instance-setup)
5. [Deploy Application](#-deploy-application)
6. [Troubleshooting](#-troubleshooting)
7. [CI/CD with GitHub Actions](#-cicd-with-github-actions)
8. [Frontend Integration](#-frontend-integration)

---

## 📦 Prerequisites

Before starting, ensure you have:

- [ ] **AWS Account** with billing enabled
- [ ] **Key Pair**: Create or reuse existing key pair
- [ ] **Local Tools**:
  - Docker Desktop installed
  - AWS CLI installed (`aws --version`)

### Local Testing First

Test your Docker setup locally before deploying:

```powershell
# Build and run locally
docker compose build
docker compose up -d

# Test the API
curl http://localhost:8000/health

# Check logs
docker compose logs -f

# Stop
docker compose down
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         INTERNET                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Cloud (ap-southeast-1)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Public Subnet                                 │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │            EC2 Instance (m7i-flex.large)             │  │  │
│  │  │  ┌───────────────┐  ┌─────────────────────────────┐ │  │  │
│  │  │  │    Nginx      │  │   MSL API (Docker)          │ │  │  │
│  │  │  │     :80       │──│      :8000                  │ │  │  │
│  │  │  │               │  │                             │ │  │  │
│  │  │  │  - WebSocket  │  │  - FastAPI                  │ │  │  │
│  │  │  │  - Rate Limit │  │  - MediaPipe                │ │  │  │
│  │  │  │               │  │  - TensorFlow               │ │  │  │
│  │  │  └───────────────┘  └─────────────────────────────┘ │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                        ECR Registry                          │ │
│  │                   msl-recognition-api:latest                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

Your Next.js App (Vercel) ──────HTTP POST──────────► EC2 /predict-image/
                          ──────WebSocket────────────► EC2 /ws/recognize
```

---

## ☁️ AWS Setup

### 1. Create ECR Repository

```bash
# Create ECR repository for MSL
aws ecr create-repository \
    --repository-name msl-recognition-api \
    --region ap-southeast-1 \
    --image-scanning-configuration scanOnPush=true

# Get the repository URI (save this!)
aws ecr describe-repositories \
    --repository-names msl-recognition-api \
    --query 'repositories[0].repositoryUri' \
    --output text
```

**Example Output:** `YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/msl-recognition-api`

### 2. Launch EC2 Instance

1. Go to **EC2 Dashboard** → **Launch Instance**
2. Configure:

   - **Name**: `msl-api-server`
   - **AMI**: Amazon Linux 2023
   - **Instance Type**: `m7i-flex.large` (2 vCPU, 8GB RAM)
   - **Key Pair**: Select or create a key pair
   - **Network Settings**:
     - ✅ **Select existing security group** or create new with ports 22, 80 open
   - **Storage**: 30GB gp3

3. Click **Launch Instance**

> **Note:** Multiple EC2 instances can share the same security group. AWS applies the rules to all associated instances.

---

## 🖥️ EC2 Instance Setup

### 1. Connect to Your Instance

```bash
# SSH into your instance
ssh -i your-key.pem ec2-user@YOUR_EC2_PUBLIC_IP
```

### 2. Install Docker

```bash
# Update system
sudo dnf update -y

# Install Docker
sudo dnf install docker -y

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose plugin
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# Log out and back in for group changes
exit
```

### 3. Configure AWS CLI on EC2

```bash
# Reconnect
ssh -i your-key.pem ec2-user@YOUR_EC2_PUBLIC_IP

# Install AWS CLI (if not present)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI
aws configure
# Enter your Access Key ID, Secret Access Key, region (ap-southeast-1), and output format (json)
```

---

## 🚀 Deploy Application

### From Your Local Machine

```powershell
# Login to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com

# Build the image
docker build -t msl-recognition-api .

# Tag for ECR
docker tag msl-recognition-api:latest YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/msl-recognition-api:latest

# Push to ECR
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/msl-recognition-api:latest
```

### On EC2 Instance

```bash
# Create application directory
mkdir -p ~/msl-api
cd ~/msl-api

# Login to ECR
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com
```

Create `docker-compose.yml`:

```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    image: YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/msl-recognition-api:latest
    container_name: msl-api
    restart: unless-stopped
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
      - PYTHONUNBUFFERED=1
      - ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    networks:
      - msl-network
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 3G
        reservations:
          cpus: '0.5'
          memory: 1G

  nginx:
    image: nginx:alpine
    container_name: msl-nginx
    restart: unless-stopped
    ports:
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      api:
        condition: service_healthy
    networks:
      - msl-network

networks:
  msl-network:
    driver: bridge
EOF
```

Create Nginx config:

```bash
mkdir -p nginx

cat > nginx/nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent"';
    access_log /var/log/nginx/access.log main;

    sendfile on;
    keepalive_timeout 65;
    client_max_body_size 10M;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    # WebSocket upgrade map
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    upstream msl_api {
        server api:8000;
        keepalive 32;
    }

    server {
        listen 80;
        server_name _;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;

        location /health {
            proxy_pass http://msl_api/health;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
        }

        location /predict-image/ {
            limit_req zone=api_limit burst=20 nodelay;

            proxy_pass http://msl_api;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws/ {
            proxy_pass http://msl_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection $connection_upgrade;
            proxy_set_header Host $host;
            proxy_read_timeout 86400;
        }

        location / {
            proxy_pass http://msl_api;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF
```

Deploy:

```bash
# Pull and run
docker compose pull
docker compose up -d

# Check status
docker compose ps
docker compose logs -f
```

### Verify Deployment

```bash
# Health check
curl http://localhost/health

# From your local machine
curl http://YOUR_EC2_PUBLIC_IP/health

# Expected response:
# {"status":"healthy","model_loaded":true,"service":"msl-recognition-api"}
```

---

## 🔧 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs api

# Check if models exist
docker compose exec api ls -la /app/models/

# Verify image
docker images
```

### ECR Login Issues

```bash
# Re-authenticate
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com

# Check credentials
aws sts get-caller-identity
```

### Out of Memory

```bash
# Check memory usage
free -h
docker stats

# Increase swap (if needed)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Useful Commands

```bash
# Restart all services
docker compose restart

# Force rebuild
docker compose up -d --force-recreate

# Remove all and start fresh
docker compose down -v
docker system prune -a
docker compose up -d

# Check resource usage
htop
docker stats
```

---

## 🔄 CI/CD with GitHub Actions

Automate deployments so pushing to `main` branch automatically deploys to EC2.

### 1. Add GitHub Secrets (Sensitive Data)

Go to your repository → **Settings** → **Secrets and variables** → **Actions** → **Secrets** tab

| Secret Name             | Value                                    |
| ----------------------- | ---------------------------------------- |
| `AWS_ACCESS_KEY_ID`     | Your IAM access key                      |
| `AWS_SECRET_ACCESS_KEY` | Your IAM secret key                      |
| `EC2_SSH_KEY`           | Contents of your `.pem` private key file |

### 2. Add GitHub Variables (Non-Sensitive Config)

Go to **Variables** tab (next to Secrets)

| Variable Name  | Value                                                  |
| -------------- | ------------------------------------------------------ |
| `AWS_REGION`   | `ap-southeast-1`                                       |
| `ECR_REGISTRY` | `YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com` |
| `EC2_HOST`     | Your EC2 public IP                                     |

### 3. Create Workflow File

Create `.github/workflows/deploy.yml` (already included in this repo).

### 4. Add SSH Key to GitHub Secrets

1. Open your `.pem` file in a text editor
2. Copy the **entire contents** including `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----`
3. In GitHub → Secrets → Add `EC2_SSH_KEY` → Paste the key contents

### 5. Deploy Workflow

After setup, every push to `main` will:

1. ✅ Build Docker image
2. ✅ Push to ECR with commit SHA tag + `latest` tag
3. ✅ SSH to EC2 and pull new image
4. ✅ Restart containers
5. ✅ Health check to verify deployment

### Manual Trigger

You can also manually trigger a deployment:

1. Go to **Actions** tab in GitHub
2. Select **Deploy MSL API to AWS**
3. Click **Run workflow** → **Run workflow**

---

## 🔗 Frontend Integration

After deploying, update your frontend:

1. **Add `MSL_API_URL` environment variable** in Vercel Dashboard:

   - Key: `MSL_API_URL`
   - Value: `http://YOUR_MSL_EC2_IP`

2. **Update `.env.local`** for local development:

   ```
   MSL_API_URL=http://YOUR_MSL_EC2_IP
   ```

3. **Redeploy** your Vercel app to pick up the new environment variable.
