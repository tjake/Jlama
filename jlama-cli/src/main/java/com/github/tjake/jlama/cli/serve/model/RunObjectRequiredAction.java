package com.github.tjake.jlama.cli.serve.model;

import com.fasterxml.jackson.annotation.JsonTypeName;
import com.github.tjake.jlama.cli.serve.model.RunObjectRequiredActionSubmitToolOutputs;
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
 * Details on the action required to continue the run. Will be &#x60;null&#x60; if no action is required.
 **/
@ApiModel(description = "Details on the action required to continue the run. Will be `null` if no action is required.")
@JsonTypeName("RunObject_required_action")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class RunObjectRequiredAction   {
  public enum TypeEnum {

    SUBMIT_TOOL_OUTPUTS(String.valueOf("submit_tool_outputs"));


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
  private @Valid RunObjectRequiredActionSubmitToolOutputs submitToolOutputs;

  /**
   * For now, this is always &#x60;submit_tool_outputs&#x60;.
   **/
  public RunObjectRequiredAction type(TypeEnum type) {
    this.type = type;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "For now, this is always `submit_tool_outputs`.")
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
   **/
  public RunObjectRequiredAction submitToolOutputs(RunObjectRequiredActionSubmitToolOutputs submitToolOutputs) {
    this.submitToolOutputs = submitToolOutputs;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("submit_tool_outputs")
  @NotNull
  public RunObjectRequiredActionSubmitToolOutputs getSubmitToolOutputs() {
    return submitToolOutputs;
  }

  @JsonProperty("submit_tool_outputs")
  public void setSubmitToolOutputs(RunObjectRequiredActionSubmitToolOutputs submitToolOutputs) {
    this.submitToolOutputs = submitToolOutputs;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    RunObjectRequiredAction runObjectRequiredAction = (RunObjectRequiredAction) o;
    return Objects.equals(this.type, runObjectRequiredAction.type) &&
        Objects.equals(this.submitToolOutputs, runObjectRequiredAction.submitToolOutputs);
  }

  @Override
  public int hashCode() {
    return Objects.hash(type, submitToolOutputs);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class RunObjectRequiredAction {\n");
    
    sb.append("    type: ").append(toIndentedString(type)).append("\n");
    sb.append("    submitToolOutputs: ").append(toIndentedString(submitToolOutputs)).append("\n");
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

