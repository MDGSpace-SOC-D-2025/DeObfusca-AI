FROM python:3.11-slim

# Install OpenJDK for Ghidra
RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Download and install Ghidra
ENV GHIDRA_VERSION=10.4_PUBLIC
ENV GHIDRA_INSTALL_PATH=/opt/ghidra

RUN wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.4_build/ghidra_10.4_PUBLIC_20230928.zip -O /tmp/ghidra.zip && \
    unzip /tmp/ghidra.zip -d /opt && \
    mv /opt/ghidra_${GHIDRA_VERSION} ${GHIDRA_INSTALL_PATH} && \
    rm /tmp/ghidra.zip

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["python", "app.py"]
