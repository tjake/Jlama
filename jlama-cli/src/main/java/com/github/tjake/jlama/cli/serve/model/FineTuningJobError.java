package com.github.tjake.jlama.cli.serve.model;

import com.fasterxml.jackson.annotation.JsonTypeName;
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

/**
 * For fine-tuning jobs that have &#x60;failed&#x60;, this will contain more information on the cause of the failure.
 **/
@ApiModel(description = "For fine-tuning jobs that have `failed`, this will contain more information on the cause of the failure.")
@JsonTypeName("FineTuningJob_error")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class FineTuningJobError   {
  private @Valid String code;
  private @Valid String message;
  private @Valid String param;

  /**
   * A machine-readable error code.
   **/
  public FineTuningJobError code(String code) {
    this.code = code;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "A machine-readable error code.")
  @JsonProperty("code")
  @NotNull
  public String getCode() {
    return code;
  }

  @JsonProperty("code")
  public void setCode(String code) {
    this.code = code;
  }

  /**
   * A human-readable error message.
   **/
  public FineTuningJobError message(String message) {
    this.message = message;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "A human-readable error message.")
  @JsonProperty("message")
  @NotNull
  public String getMessage() {
    return message;
  }

  @JsonProperty("message")
  public void setMessage(String message) {
    this.message = message;
  }

  /**
   * The parameter that was invalid, usually &#x60;training_file&#x60; or &#x60;validation_file&#x60;. This field will be null if the failure was not parameter-specific.
   **/
  public FineTuningJobError param(String param) {
    this.param = param;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The parameter that was invalid, usually `training_file` or `validation_file`. This field will be null if the failure was not parameter-specific.")
  @JsonProperty("param")
  @NotNull
  public String getParam() {
    return param;
  }

  @JsonProperty("param")
  public void setParam(String param) {
    this.param = param;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    FineTuningJobError fineTuningJobError = (FineTuningJobError) o;
    return Objects.equals(this.code, fineTuningJobError.code) &&
        Objects.equals(this.message, fineTuningJobError.message) &&
        Objects.equals(this.param, fineTuningJobError.param);
  }

  @Override
  public int hashCode() {
    return Objects.hash(code, message, param);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class FineTuningJobError {\n");
    
    sb.append("    code: ").append(toIndentedString(code)).append("\n");
    sb.append("    message: ").append(toIndentedString(message)).append("\n");
    sb.append("    param: ").append(toIndentedString(param)).append("\n");
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

