package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.SubmitToolOutputsRunRequestToolOutputsInner;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jakarta.validation.constraints.*;
import jakarta.validation.Valid;

import io.swagger.annotations.*;
import java.util.Objects;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import com.fasterxml.jackson.annotation.JsonTypeName;



@JsonTypeName("SubmitToolOutputsRunRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class SubmitToolOutputsRunRequest   {
  private @Valid List<@Valid SubmitToolOutputsRunRequestToolOutputsInner> toolOutputs = new ArrayList<>();

  /**
   * A list of tools for which the outputs are being submitted.
   **/
  public SubmitToolOutputsRunRequest toolOutputs(List<@Valid SubmitToolOutputsRunRequestToolOutputsInner> toolOutputs) {
    this.toolOutputs = toolOutputs;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "A list of tools for which the outputs are being submitted.")
  @JsonProperty("tool_outputs")
  @NotNull
  public List<SubmitToolOutputsRunRequestToolOutputsInner> getToolOutputs() {
    return toolOutputs;
  }

  @JsonProperty("tool_outputs")
  public void setToolOutputs(List<@Valid SubmitToolOutputsRunRequestToolOutputsInner> toolOutputs) {
    this.toolOutputs = toolOutputs;
  }

  public SubmitToolOutputsRunRequest addToolOutputsItem(SubmitToolOutputsRunRequestToolOutputsInner toolOutputsItem) {
    if (this.toolOutputs == null) {
      this.toolOutputs = new ArrayList<>();
    }

    this.toolOutputs.add(toolOutputsItem);
    return this;
  }

  public SubmitToolOutputsRunRequest removeToolOutputsItem(SubmitToolOutputsRunRequestToolOutputsInner toolOutputsItem) {
    if (toolOutputsItem != null && this.toolOutputs != null) {
      this.toolOutputs.remove(toolOutputsItem);
    }

    return this;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    SubmitToolOutputsRunRequest submitToolOutputsRunRequest = (SubmitToolOutputsRunRequest) o;
    return Objects.equals(this.toolOutputs, submitToolOutputsRunRequest.toolOutputs);
  }

  @Override
  public int hashCode() {
    return Objects.hash(toolOutputs);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class SubmitToolOutputsRunRequest {\n");
    
    sb.append("    toolOutputs: ").append(toIndentedString(toolOutputs)).append("\n");
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

