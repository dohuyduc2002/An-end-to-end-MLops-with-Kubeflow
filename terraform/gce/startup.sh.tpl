#!/bin/bash
set -e

# Install system utilities
apt-get update
apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  software-properties-common \
  gnupg \
  lsb-release \
  python3-pip \
  virtualenv \
  python3-setuptools \
  nginx

# Install Docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable docker
systemctl restart docker

# Append DNS mock entries to /etc/hosts
%{ for entry in dns_hosts ~}
echo "${entry.ip} ${entry.hostname}" >> /etc/hosts
%{ endfor ~}

# Pull and run Jenkins container
docker pull ${jenkins_image}
docker run -d \
  --name jenkins \
  -p 8080:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  ${jenkins_image}

# Configure NGINX reverse proxy to Jenkins
cat <<EOF > /etc/nginx/sites-available/jenkins
server {
    listen 80 default_server;

    location / {
        proxy_pass         http://localhost:8080;
        proxy_set_header   Host \$host;
        proxy_set_header   X-Real-IP \$remote_addr;
        proxy_set_header   X-Forwarded-For \$proxy_add_x_forwarded_for;
    }
}
EOF

# Enable NGINX site and restart service
ln -s /etc/nginx/sites-available/jenkins /etc/nginx/sites-enabled/jenkins
rm -f /etc/nginx/sites-enabled/default
systemctl restart nginx
