FROM ubuntu:22.04 AS builder
RUN apt-get update
RUN apt-get install -y build-essential git zip curl zlib1g-dev

ENV SDKMAN_DIR=/root/.sdkman
ENV JAVA_VERSION_20=20.0.2-graalce
ENV JAVA_VERSION_21=21.0.2-graalce
ENV JAVA_VERSION_22=22.0.2-graalce

RUN ["mkdir", "-p", "/build"]

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN curl -s "https://get.sdkman.io" | bash
RUN chmod a+x "$SDKMAN_DIR/bin/sdkman-init.sh"

RUN set -x \
    && echo "sdkman_auto_answer=true" > $SDKMAN_DIR/etc/config \
    && echo "sdkman_auto_selfupdate=false" >> $SDKMAN_DIR/etc/config \
    && echo "sdkman_insecure_ssl=false" >> $SDKMAN_DIR/etc/config

WORKDIR $SDKMAN_DIR
RUN [[ -s "$SDKMAN_DIR/bin/sdkman-init.sh" ]] && source "$SDKMAN_DIR/bin/sdkman-init.sh" && exec "$@"

RUN source /root/.bashrc
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install java $JAVA_VERSION_20
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install java $JAVA_VERSION_21
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install java $JAVA_VERSION_22

WORKDIR /build
RUN git clone https://github.com/tjake/sdkman-for-toolchains.git
WORKDIR /build/sdkman-for-toolchains
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && ./mvnw -Pnative clean package
RUN ["mkdir", "-p", "/root/.m2"]
RUN printf "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<toolchains>\n</toolchains>" > /root/.m2/toolchains.xml
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && target/toolchains generate
RUN cat  /root/.m2/toolchains.xml
WORKDIR /build
COPY jlama-cli jlama-cli
COPY jlama-core jlama-core
COPY jlama-native jlama-native
COPY jlama-net jlama-net
COPY jlama-tests jlama-tests
COPY .mvn .mvn
COPY pom.xml .
COPY mvnw .

RUN --mount=type=cache,target=/root/.m2/repository source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk use java $JAVA_VERSION_22 && ./mvnw clean package -DskipTests

FROM openjdk:21-slim
RUN apt-get update
RUN apt-get install -y procps curl gzip

LABEL org.opencontainers.image.source=https://github.com/tjake/Jlama
RUN mkdir -p /profiler && curl -s -L https://github.com/async-profiler/async-profiler/releases/download/v3.0/async-profiler-3.0-linux-x64.tar.gz | tar zxvf - -C /profiler

COPY inlinerules.json inlinerules.json
COPY run-cli.sh run-cli.sh
COPY conf/logback.xml logback.xml
COPY --from=builder /build/jlama-cli/target/jlama-cli.jar ./jlama-cli.jar

ENV JLAMA_PREINSTALLED_JAR=/jlama-cli.jar
ENV JLAMA_JVM_ARGS="-Dlogback.configurationFile=./logback.xml"

ENTRYPOINT ["./run-cli.sh"]