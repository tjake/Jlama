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



@JsonTypeName("SubmitToolOutputsRunRequest_tool_outputs_inner")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class SubmitToolOutputsRunRequestToolOutputsInner   {
  private @Valid String toolCallId;
  private @Valid String output;

  /**
   * The ID of the tool call in the &#x60;required_action&#x60; object within the run object the output is being submitted for.
   **/
  public SubmitToolOutputsRunRequestToolOutputsInner toolCallId(String toolCallId) {
    this.toolCallId = toolCallId;
    return this;
  }

  
  @ApiModelProperty(value = "The ID of the tool call in the `required_action` object within the run object the output is being submitted for.")
  @JsonProperty("tool_call_id")
  public String getToolCallId() {
    return toolCallId;
  }

  @JsonProperty("tool_call_id")
  public void setToolCallId(String toolCallId) {
    this.toolCallId = toolCallId;
  }

  /**
   * The output of the tool call to be submitted to continue the run.
   **/
  public SubmitToolOutputsRunRequestToolOutputsInner output(String output) {
    this.output = output;
    return this;
  }

  
  @ApiModelProperty(value = "The output of the tool call to be submitted to continue the run.")
  @JsonProperty("output")
  public String getOutput() {
    return output;
  }

  @JsonProperty("output")
  public void setOutput(String output) {
    this.output = output;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    SubmitToolOutputsRunRequestToolOutputsInner submitToolOutputsRunRequestToolOutputsInner = (SubmitToolOutputsRunRequestToolOutputsInner) o;
    return Objects.equals(this.toolCallId, submitToolOutputsRunRequestToolOutputsInner.toolCallId) &&
        Objects.equals(this.output, submitToolOutputsRunRequestToolOutputsInner.output);
  }

  @Override
  public int hashCode() {
    return Objects.hash(toolCallId, output);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class SubmitToolOutputsRunRequestToolOutputsInner {\n");
    
    sb.append("    toolCallId: ").append(toIndentedString(toolCallId)).append("\n");
    sb.append("    output: ").append(toIndentedString(output)).append("\n");
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

