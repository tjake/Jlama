#!/bin/bash

# Function to get Java executable path
get_java_exec() {
  if [[ -n "$JAVA_HOME" ]]; then
    echo "$JAVA_HOME/bin/java"
  else
    echo "java"
  fi
}

# Function to extract the major version of Java
get_java_major_version() {
  local version=$($1 -version 2>&1 | awk -F '"' '/version/ {print $2}')
  echo ${version%%.*}
}

# Verify Java version is JDK 20/21/22
JAVA=$(get_java_exec)
JAVA_MAJOR_VERSION=$(get_java_major_version $JAVA)
if [[ "$JAVA_MAJOR_VERSION" != "20" ]] && [[ "$JAVA_MAJOR_VERSION" != "21" ]] && [[ "$JAVA_MAJOR_VERSION" != "22" ]]; then
  echo "Error: JDK 20/21/22 is required to run this application."
  exit 1
fi

# Define the path of the relative JAR
JLAMA_RELATIVE_JAR="./jlama-cli/target/jlama-cli.jar"
# Path to the logback.xml
LOGBACK_CONFIG="./conf/logback2.xml"

JLAMA_JVM_ARGS="$JLAMA_JVM_ARGS -server -Xmx12G -Dstdout.encoding=UTF-8 -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 --add-modules=jdk.incubator.vector --add-exports java.base/sun.nio.ch=ALL-UNNAMED --enable-preview --enable-native-access=ALL-UNNAMED \
 -XX:+UnlockDiagnosticVMOptions -XX:CompilerDirectivesFile=./inlinerules.json -XX:+AlignVector -XX:+UseStringDeduplication \
 -XX:+UseCompressedOops -XX:+UseCompressedClassPointers "
        
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
  $JAVA $JLAMA_JVM_ARGS $JLAMA_JVM_ARGS_EXTRA -Dlogback.configurationFile=$LOGBACK_CONFIG -jar $JLAMA_RELATIVE_JAR "$@"
else
  # If PREINSTALLED_JAR is set, run the JAR specified by the variable
  $JAVA $JLAMA_JVM_ARGS $JLAMA_JVM_ARGS_EXTRA -jar $JLAMA_PREINSTALLED_JAR "$@"
fi
