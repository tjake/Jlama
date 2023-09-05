#!/bin/bash

set -x

# GraalVM version 
VERSION=20.0.0

# Directory to install GraalVM
GRAAL_HOME=/opt/graalvm-$VERSION

# Download GraalVM CE jar 
curl -OL https://github.com/graalvm/graalvm-ce-builds/releases/download/jdk-20.0.2/graalvm-community-jdk-20.0.2_linux-x64_bin.tar.gz

# Extract GraalVM
tar -xvf graalvm-community-jdk-20.0.2_linux-x64_bin.tar.gz

user=$(whoami)
# Move files to installation directory
sudo mkdir -p ${GRAAL_HOME}
sudo chown ${user}:${user} ${GRAAL_HOME}

mv graalvm-community-openjdk-20.0.2+9.1/* ${GRAAL_HOME}

# Set environment variables
export GRAALVM_HOME=${GRAAL_HOME}
export PATH=${GRAAL_HOME}/bin:$PATH

echo "GraalVM installed at: $GRAAL_HOME"

export JAVA_HOME=${GRAAL_HOME}
# Verify installation
${GRAAL_HOME}/bin/java -version
