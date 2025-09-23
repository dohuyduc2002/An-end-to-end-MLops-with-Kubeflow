FROM unitycatalog/unitycatalog:v0.3.0

USER root

# Install dependencies
RUN apk add --no-cache python3 py3-pip curl tar bash libc6-compat

# Create Python venv
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Google Cloud CLI
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-465.0.0-linux-x86_64.tar.gz \
    && tar -xf google-cloud-cli-465.0.0-linux-x86_64.tar.gz \
    && mv google-cloud-sdk /opt/google-cloud-sdk \
    && /opt/google-cloud-sdk/install.sh --quiet \
    && rm google-cloud-cli-465.0.0-linux-x86_64.tar.gz

# Add gcloud and Python venv to PATH
ENV PATH="/opt/venv/bin:/opt/google-cloud-sdk/bin:$PATH"

USER unitycatalog
WORKDIR /home/unitycatalog
