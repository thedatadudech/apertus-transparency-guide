# 🚀 SSH Server Deployment Guide

## Deploying Apertus-8B on Remote GPU Server with SSH Access

This guide shows how to deploy Apertus Swiss AI on a remote GPU server and access it locally via SSH tunneling.

---

## 🎯 Prerequisites

- **Remote GPU Server** with CUDA support (A40, A100, RTX 4090, etc.)
- **SSH access** to the server
- **Hugging Face access** to `swiss-ai/Apertus-8B-Instruct-2509`
- **Local machine** for accessing the dashboard

---

## 📦 Server Setup

### 1. Connect to Your Server

```bash
ssh username@your-server-ip
# Or if using a specific key:
ssh -i your-key.pem username@your-server-ip
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/apertus-transparency-guide.git
cd apertus-transparency-guide
```

### 3. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers accelerate
pip install -r requirements.txt

# Install package
pip install -e .
```

### 4. Authenticate with Hugging Face

```bash
# Login to Hugging Face (required for model access)
huggingface-cli login
# Enter your token when prompted
```

### 5. Verify GPU Setup

```bash
# Check GPU availability
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## 🔧 Running Applications

### Option 1: Basic Chat Interface

```bash
# Run basic chat directly on server
python examples/basic_chat.py
```

### Option 2: Streamlit Dashboard with Port Forwarding

#### Start Streamlit on Server

```bash
# On your remote server
streamlit run dashboards/streamlit_transparency.py --server.port 8501 --server.address 0.0.0.0
```

#### Setup SSH Port Forwarding (From Local Machine)

```bash
# From your local machine, create SSH tunnel
ssh -L 8501:localhost:8501 username@your-server-ip

# Or with specific key:
ssh -L 8501:localhost:8501 -i your-key.pem username@your-server-ip
```

#### Access Dashboard Locally

Open your local browser and go to:
```
http://localhost:8501
```

The Streamlit dashboard will now be accessible on your local machine!

### Option 3: vLLM API Server

#### Start vLLM Server

```bash
# On your remote server
python -m vllm.entrypoints.openai.api_server \
    --model swiss-ai/Apertus-8B-Instruct-2509 \
    --dtype bfloat16 \
    --temperature 0.8 \
    --top-p 0.9 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000
```

#### Setup Port Forwarding for API

```bash
# From local machine
ssh -L 8000:localhost:8000 username@your-server-ip
```

#### Test API Locally

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="swiss-ai/Apertus-8B-Instruct-2509",
    messages=[{"role": "user", "content": "Hello from remote server!"}],
    temperature=0.8
)

print(response.choices[0].message.content)
```

---

## 🛠️ Advanced Configuration

### Multiple Port Forwarding

You can forward multiple services at once:

```bash
# Forward both Streamlit (8501) and vLLM API (8000)
ssh -L 8501:localhost:8501 -L 8000:localhost:8000 username@your-server-ip
```

### Background Process Management

#### Using Screen (Recommended)

```bash
# Start a screen session
screen -S apertus

# Run your application inside screen
streamlit run dashboards/streamlit_transparency.py --server.port 8501 --server.address 0.0.0.0

# Detach: Ctrl+A, then D
# Reattach: screen -r apertus
# List sessions: screen -ls
```

#### Using nohup

```bash
# Run in background with nohup
nohup streamlit run dashboards/streamlit_transparency.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

# Check if running
ps aux | grep streamlit

# View logs
tail -f streamlit.log
```

#### Using systemd (Production)

Create service file:

```bash
sudo nano /etc/systemd/system/apertus-dashboard.service
```

```ini
[Unit]
Description=Apertus Transparency Dashboard
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/apertus-transparency-guide
Environment=PATH=/path/to/apertus-transparency-guide/.venv/bin
ExecStart=/path/to/apertus-transparency-guide/.venv/bin/streamlit run dashboards/streamlit_transparency.py --server.port 8501 --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable apertus-dashboard
sudo systemctl start apertus-dashboard

# Check status
sudo systemctl status apertus-dashboard
```

---

## 🔒 Security Considerations

### SSH Key Authentication

Always use SSH keys instead of passwords:

```bash
# Generate key pair (on local machine)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/apertus_server

# Copy public key to server
ssh-copy-id -i ~/.ssh/apertus_server.pub username@your-server-ip

# Connect with key
ssh -i ~/.ssh/apertus_server username@your-server-ip
```

### Firewall Configuration

```bash
# Only allow SSH and your specific ports
sudo ufw allow ssh
sudo ufw allow from your-local-ip to any port 8501
sudo ufw allow from your-local-ip to any port 8000
sudo ufw enable
```

### SSH Config

Create `~/.ssh/config` on your local machine:

```
Host apertus
    HostName your-server-ip
    User your-username
    IdentityFile ~/.ssh/apertus_server
    LocalForward 8501 localhost:8501
    LocalForward 8000 localhost:8000
```

Then simply connect with:

```bash
ssh apertus
```

---

## 📊 Performance Monitoring

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Or install nvtop for better interface
sudo apt install nvtop
nvtop
```

### System Monitoring

```bash
# System resources
htop

# Or install and use btop
sudo apt install btop
btop
```

### Application Monitoring

```bash
# Monitor Streamlit process
ps aux | grep streamlit

# Check logs
journalctl -u apertus-dashboard -f  # for systemd service
tail -f streamlit.log               # for nohup
```

---

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Fails

```bash
# Check HuggingFace authentication
huggingface-cli whoami

# Clear cache and retry
rm -rf ~/.cache/huggingface/
huggingface-cli login
```

#### Out of GPU Memory

```bash
# Check GPU memory usage
nvidia-smi

# Consider using quantization
python examples/basic_chat.py --load-in-8bit
```

#### Port Already in Use

```bash
# Find what's using the port
sudo lsof -i :8501

# Kill process if needed
sudo kill -9 <PID>
```

#### SSH Connection Issues

```bash
# Test connection
ssh -v username@your-server-ip

# Check if port forwarding is working
netstat -tlnp | grep 8501
```

### Logs and Debugging

```bash
# Check system logs
sudo journalctl -xe

# Check SSH daemon logs
sudo journalctl -u ssh

# Debug Streamlit issues
streamlit run dashboards/streamlit_transparency.py --logger.level debug
```

---

## 🚀 Quick Commands Reference

```bash
# Connect with port forwarding
ssh -L 8501:localhost:8501 username@your-server-ip

# Start Streamlit dashboard
streamlit run dashboards/streamlit_transparency.py --server.port 8501 --server.address 0.0.0.0

# Start vLLM API server
python -m vllm.entrypoints.openai.api_server --model swiss-ai/Apertus-8B-Instruct-2509 --host 0.0.0.0 --port 8000

# Monitor GPU
nvidia-smi

# Check running processes
ps aux | grep -E "(streamlit|vllm)"
```

Mit dieser Anleitung kannst du Apertus auf deinem GPU-Server laufen lassen und lokal über SSH-Port-Forwarding darauf zugreifen! 🇨🇭