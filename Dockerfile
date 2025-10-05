FROM eclipse-temurin:21-jdk-jammy AS builder

RUN cat /etc/os-release
RUN apt-get update
RUN apt-get install -y build-essential git zip curl zlib1g-dev jq

WORKDIR /build

COPY jlama-cli jlama-cli
COPY jlama-core jlama-core
COPY jlama-native jlama-native
COPY jlama-net jlama-net
COPY jlama-tests jlama-tests
COPY .mvn .mvn
COPY pom.xml .
COPY mvnw .
COPY .git .git

RUN ./mvnw clean package -DskipTests

FROM eclipse-temurin:21-jre-alpine
RUN apk update
RUN apk add procps curl gzip

LABEL org.opencontainers.image.source=https://github.com/tjake/Jlama
RUN mkdir -p /profiler && curl -s -L https://github.com/async-profiler/async-profiler/releases/download/v3.0/async-profiler-3.0-linux-x64.tar.gz | tar zxvf - -C /profiler

COPY inlinerules.json inlinerules.json
COPY run-cli.sh run-cli.sh
COPY conf/logback.xml logback.xml
COPY --from=builder /build/jlama-cli/target/jlama-cli.jar ./jlama-cli.jar

ENV JLAMA_PREINSTALLED_JAR=/jlama-cli.jar
ENV JLAMA_JVM_ARGS="-Dlogback.configurationFile=./logback.xml"

ENTRYPOINT ["./run-cli.sh"]