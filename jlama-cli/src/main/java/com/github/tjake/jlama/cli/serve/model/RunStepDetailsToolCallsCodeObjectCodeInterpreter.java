package com.github.tjake.jlama.cli.serve.model;

import com.fasterxml.jackson.annotation.JsonTypeName;
import com.github.tjake.jlama.cli.serve.model.RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner;
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

/**
 * The Code Interpreter tool call definition.
 **/
@ApiModel(description = "The Code Interpreter tool call definition.")
@JsonTypeName("RunStepDetailsToolCallsCodeObject_code_interpreter")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class RunStepDetailsToolCallsCodeObjectCodeInterpreter   {
  private @Valid String input;
  private @Valid List<@Valid RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner> outputs = new ArrayList<>();

  /**
   * The input to the Code Interpreter tool call.
   **/
  public RunStepDetailsToolCallsCodeObjectCodeInterpreter input(String input) {
    this.input = input;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The input to the Code Interpreter tool call.")
  @JsonProperty("input")
  @NotNull
  public String getInput() {
    return input;
  }

  @JsonProperty("input")
  public void setInput(String input) {
    this.input = input;
  }

  /**
   * The outputs from the Code Interpreter tool call. Code Interpreter can output one or more items, including text (&#x60;logs&#x60;) or images (&#x60;image&#x60;). Each of these are represented by a different object type.
   **/
  public RunStepDetailsToolCallsCodeObjectCodeInterpreter outputs(List<@Valid RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner> outputs) {
    this.outputs = outputs;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The outputs from the Code Interpreter tool call. Code Interpreter can output one or more items, including text (`logs`) or images (`image`). Each of these are represented by a different object type.")
  @JsonProperty("outputs")
  @NotNull
  public List<RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner> getOutputs() {
    return outputs;
  }

  @JsonProperty("outputs")
  public void setOutputs(List<@Valid RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner> outputs) {
    this.outputs = outputs;
  }

  public RunStepDetailsToolCallsCodeObjectCodeInterpreter addOutputsItem(RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner outputsItem) {
    if (this.outputs == null) {
      this.outputs = new ArrayList<>();
    }

    this.outputs.add(outputsItem);
    return this;
  }

  public RunStepDetailsToolCallsCodeObjectCodeInterpreter removeOutputsItem(RunStepDetailsToolCallsCodeObjectCodeInterpreterOutputsInner outputsItem) {
    if (outputsItem != null && this.outputs != null) {
      this.outputs.remove(outputsItem);
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
    RunStepDetailsToolCallsCodeObjectCodeInterpreter runStepDetailsToolCallsCodeObjectCodeInterpreter = (RunStepDetailsToolCallsCodeObjectCodeInterpreter) o;
    return Objects.equals(this.input, runStepDetailsToolCallsCodeObjectCodeInterpreter.input) &&
        Objects.equals(this.outputs, runStepDetailsToolCallsCodeObjectCodeInterpreter.outputs);
  }

  @Override
  public int hashCode() {
    return Objects.hash(input, outputs);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class RunStepDetailsToolCallsCodeObjectCodeInterpreter {\n");
    
    sb.append("    input: ").append(toIndentedString(input)).append("\n");
    sb.append("    outputs: ").append(toIndentedString(outputs)).append("\n");
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

