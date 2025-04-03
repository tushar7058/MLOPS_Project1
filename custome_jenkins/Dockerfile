# Use the official Jenkins image as base
FROM jenkins/jenkins:lts

# Switch to root user
USER root

# Install prerequisites and Docker
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        apt-transport-https ca-certificates curl gnupg software-properties-common && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | tee /usr/share/keyrings/docker-archive-keyring.gpg > /dev/null && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian bullseye stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io && \
    rm -rf /var/lib/apt/lists/*

# Add Jenkins user to the Docker group
RUN usermod -aG docker jenkins

# Create Docker directory and volume
RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

# Ensure Jenkins can access Docker socket at runtime
RUN chmod 666 /var/run/docker.sock || true

# Switch back to Jenkins user
USER jenkins