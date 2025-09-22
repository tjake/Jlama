
# Developer Guide

So you want to contribute code to Jlama? Excellent! We're glad you're here. Here's what you need to do.

## Getting Started

### Fork jlama Repo

Fork [tjake/jlama](https://github.com/tjake/jlama) and clone locally.

Example:
```
git clone https://github.com/[your username]/jlama.git
```

### Install Prerequisites

#### JDK 21+

Jlama development builds use Java 21. You must have a JDK 21 installed with the environment variable
`JAVA_HOME` referencing the path to Java home for your JDK 21 installation, e.g. `JAVA_HOME=/usr/lib/jvm/jdk-21`.

One easy way to get Java 21 on *nix is to use [sdkman](https://sdkman.io/).

```bash
curl -s "https://get.sdkman.io" | bash
source ~/.sdkman/bin/sdkman-init.sh
sdk install java 21.0.2-open
sdk use java 21.0.2-open
```

#### jlama-native tools

The jlama-native package builds native code for cpu/gpus

For this to work you need `make` and a modern c compiler like `clang` installed

## Build

Jlama  uses a maven wrapper for its build.
Run `mvnw` on Unix systems.

Build Jlama using `./mvnw clean package`

```bash
./mvnw clean package
```

## Run Jlama

### Run Jlama Locally
Run Jlama cli using `run-cli.sh` program.

```shell script
./run-cli.sh
```

