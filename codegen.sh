java -jar openapi-generator-cli.jar generate -i source_openapi.yaml -g jaxrs-spec  --additional-properties=useJakartaEe=true,modelPackage=com.github.tjake.jlama.cli.serve --skip-validate-spec -o ./generated

