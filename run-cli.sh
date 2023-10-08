#!/bin/bash

# Function to extract the major version of Java
get_java_major_version() {
  local version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}')
  echo ${version%%.*}
}

# Verify Java version is JDK 21
JAVA_MAJOR_VERSION=$(get_java_major_version)

if [[ "$JAVA_MAJOR_VERSION" != "21" ]]; then
  echo "Error: JDK 21 is required to run this application."
  exit 1
fi

# Define the path of the relative JAR
JLAMA_RELATIVE_JAR="./jlama-cli/target/jlama-cli.jar"
# Path to the logback.xml
LOGBACK_CONFIG="./conf/logback.xml"


JLAMA_JVM_ARGS="-server -Xmx12g --add-modules=jdk.incubator.vector --add-exports java.base/sun.nio.ch=ALL-UNNAMED --enable-preview --enable-native-access=ALL-UNNAMED \
 -XX:+UnlockDiagnosticVMOptions -XX:CompilerDirectivesFile=./inlinerules.json -XX:+AlignVector"
        
# Check if PREINSTALLED_JAR environment variable is set
if [[ -z "$JLAMA_PREINSTALLED_JAR" ]]; then
  # If the relative JAR doesn't exist, build it
  if [[ ! -f $JLAMA_RELATIVE_JAR ]]; then
    echo "The JAR $JLAMA_RELATIVE_JAR is missing. Attempting to build..."
    ./mvnw clean package -DskipTests
    if [[ $? -ne 0 ]]; then
      echo "Error building the JAR. Please check your build setup."
      exit 1
    fi
  fi
  # Run the JAR in a relative directory
  java $JLAMA_JVM_ARGS -Dlogback.configurationFile=$LOGBACK_CONFIG -jar $JLAMA_RELATIVE_JAR "$@"
else
  # If PREINSTALLED_JAR is set, run the JAR specified by the variable
  java $JLAMA_JVM_ARGS -jar $JLAMA_PREINSTALLED_JAR "$@"
fi
