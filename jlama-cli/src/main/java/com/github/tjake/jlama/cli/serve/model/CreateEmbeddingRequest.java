package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CreateEmbeddingRequestInput;
import com.github.tjake.jlama.cli.serve.model.CreateEmbeddingRequestModel;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import jakarta.validation.constraints.*;
import jakarta.validation.Valid;

import io.swagger.annotations.*;
import java.util.Objects;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import com.fasterxml.jackson.annotation.JsonTypeName;



@JsonTypeName("CreateEmbeddingRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateEmbeddingRequest   {
  private @Valid CreateEmbeddingRequestInput input;
  private @Valid CreateEmbeddingRequestModel model;
  public enum EncodingFormatEnum {

    FLOAT(String.valueOf("float")), BASE64(String.valueOf("base64"));


    private String value;

    EncodingFormatEnum (String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    @Override
    @JsonValue
    public String toString() {
        return String.valueOf(value);
    }

    /**
     * Convert a String into String, as specified in the
     * <a href="https://download.oracle.com/otndocs/jcp/jaxrs-2_0-fr-eval-spec/index.html">See JAX RS 2.0 Specification, section 3.2, p. 12</a>
     */
    public static EncodingFormatEnum fromString(String s) {
        for (EncodingFormatEnum b : EncodingFormatEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected string value '" + s + "'");
    }

    @JsonCreator
    public static EncodingFormatEnum fromValue(String value) {
        for (EncodingFormatEnum b : EncodingFormatEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }
}

  private @Valid EncodingFormatEnum encodingFormat = EncodingFormatEnum.FLOAT;

  /**
   **/
  public CreateEmbeddingRequest input(CreateEmbeddingRequestInput input) {
    this.input = input;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("input")
  @NotNull
  public CreateEmbeddingRequestInput getInput() {
    return input;
  }

  @JsonProperty("input")
  public void setInput(CreateEmbeddingRequestInput input) {
    this.input = input;
  }

  /**
   **/
  public CreateEmbeddingRequest model(CreateEmbeddingRequestModel model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("model")
  @NotNull
  public CreateEmbeddingRequestModel getModel() {
    return model;
  }

  @JsonProperty("model")
  public void setModel(CreateEmbeddingRequestModel model) {
    this.model = model;
  }

  /**
   * The format to return the embeddings in. Can be either &#x60;float&#x60; or [&#x60;base64&#x60;](https://pypi.org/project/pybase64/).
   **/
  public CreateEmbeddingRequest encodingFormat(EncodingFormatEnum encodingFormat) {
    this.encodingFormat = encodingFormat;
    return this;
  }

  
  @ApiModelProperty(example = "float", value = "The format to return the embeddings in. Can be either `float` or [`base64`](https://pypi.org/project/pybase64/).")
  @JsonProperty("encoding_format")
  public EncodingFormatEnum getEncodingFormat() {
    return encodingFormat;
  }

  @JsonProperty("encoding_format")
  public void setEncodingFormat(EncodingFormatEnum encodingFormat) {
    this.encodingFormat = encodingFormat;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateEmbeddingRequest createEmbeddingRequest = (CreateEmbeddingRequest) o;
    return Objects.equals(this.input, createEmbeddingRequest.input) &&
        Objects.equals(this.model, createEmbeddingRequest.model) &&
        Objects.equals(this.encodingFormat, createEmbeddingRequest.encodingFormat);
  }

  @Override
  public int hashCode() {
    return Objects.hash(input, model, encodingFormat);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateEmbeddingRequest {\n");
    
    sb.append("    input: ").append(toIndentedString(input)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    encodingFormat: ").append(toIndentedString(encodingFormat)).append("\n");
    sb.append("}");
    return sb.toString();
  }

  /**
   * Convert the given object to string with each line indented by 4 spaces
   * (except the first line).
   */
  private String toIndentedString(Object o) {
    if (o == null) {
      return "null";
    }
    return o.toString().replace("\n", "\n    ");
  }


}

