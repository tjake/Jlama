FROM openjdk:21-slim as builder
RUN apt-get update
RUN apt-get install -y build-essential

COPY jlama-cli jlama-cli
COPY jlama-core jlama-core
COPY jlama-native jlama-native
COPY jlama-net jlama-net
COPY jlama-tests jlama-tests
COPY .mvn .mvn
COPY pom.xml .
COPY mvnw .

RUN --mount=type=cache,target=/root/.m2 ./mvnw clean package

FROM openjdk:21-slim
RUN apt-get update
RUN apt-get install -y procps curl

LABEL org.opencontainers.image.source=https://github.com/tjake/Jlama

COPY inlinerules.json inlinerules.json
COPY run-cli.sh run-cli.sh
COPY conf/logback.xml logback.xml
COPY --from=builder jlama-cli/target/jlama-cli.jar ./jlama-cli.jar

ENV JLAMA_PREINSTALLED_JAR=/jlama-cli.jar
ENV JLAMA_JVM_ARGS="-Dlogback.configurationFile=./logback.xml"

ENTRYPOINT ["./run-cli.sh"]