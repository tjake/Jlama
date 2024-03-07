package com.github.tjake.jlama.cli.serve.model;

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
 * Text output from the Code Interpreter tool call as part of a run step.
 **/
@ApiModel(description = "Text output from the Code Interpreter tool call as part of a run step.")
@JsonTypeName("RunStepDetailsToolCallsCodeOutputLogsObject")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class RunStepDetailsToolCallsCodeOutputLogsObject   {
  public enum TypeEnum {

    LOGS(String.valueOf("logs"));


    private String value;

    TypeEnum (String v) {
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
    public static TypeEnum fromString(String s) {
        for (TypeEnum b : TypeEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected string value '" + s + "'");
    }

    @JsonCreator
    public static TypeEnum fromValue(String value) {
        for (TypeEnum b : TypeEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }
}

  private @Valid TypeEnum type;
  private @Valid String logs;

  /**
   * Always &#x60;logs&#x60;.
   **/
  public RunStepDetailsToolCallsCodeOutputLogsObject type(TypeEnum type) {
    this.type = type;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Always `logs`.")
  @JsonProperty("type")
  @NotNull
  public TypeEnum getType() {
    return type;
  }

  @JsonProperty("type")
  public void setType(TypeEnum type) {
    this.type = type;
  }

  /**
   * The text output from the Code Interpreter tool call.
   **/
  public RunStepDetailsToolCallsCodeOutputLogsObject logs(String logs) {
    this.logs = logs;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The text output from the Code Interpreter tool call.")
  @JsonProperty("logs")
  @NotNull
  public String getLogs() {
    return logs;
  }

  @JsonProperty("logs")
  public void setLogs(String logs) {
    this.logs = logs;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    RunStepDetailsToolCallsCodeOutputLogsObject runStepDetailsToolCallsCodeOutputLogsObject = (RunStepDetailsToolCallsCodeOutputLogsObject) o;
    return Objects.equals(this.type, runStepDetailsToolCallsCodeOutputLogsObject.type) &&
        Objects.equals(this.logs, runStepDetailsToolCallsCodeOutputLogsObject.logs);
  }

  @Override
  public int hashCode() {
    return Objects.hash(type, logs);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class RunStepDetailsToolCallsCodeOutputLogsObject {\n");
    
    sb.append("    type: ").append(toIndentedString(type)).append("\n");
    sb.append("    logs: ").append(toIndentedString(logs)).append("\n");
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

