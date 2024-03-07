package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CreateEditRequestModel;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import java.math.BigDecimal;

import jakarta.validation.constraints.*;
import jakarta.validation.Valid;

import io.swagger.annotations.*;
import java.util.Objects;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import com.fasterxml.jackson.annotation.JsonTypeName;



@JsonTypeName("CreateEditRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateEditRequest   {
  private @Valid String instruction;
  private @Valid CreateEditRequestModel model;
  private @Valid String input = "";
  private @Valid Integer n = 1;
  private @Valid BigDecimal temperature = new BigDecimal("1");
  private @Valid BigDecimal topP = new BigDecimal("1");

  /**
   * The instruction that tells the model how to edit the prompt.
   **/
  public CreateEditRequest instruction(String instruction) {
    this.instruction = instruction;
    return this;
  }

  
  @ApiModelProperty(example = "Fix the spelling mistakes.", required = true, value = "The instruction that tells the model how to edit the prompt.")
  @JsonProperty("instruction")
  @NotNull
  public String getInstruction() {
    return instruction;
  }

  @JsonProperty("instruction")
  public void setInstruction(String instruction) {
    this.instruction = instruction;
  }

  /**
   **/
  public CreateEditRequest model(CreateEditRequestModel model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("model")
  @NotNull
  public CreateEditRequestModel getModel() {
    return model;
  }

  @JsonProperty("model")
  public void setModel(CreateEditRequestModel model) {
    this.model = model;
  }

  /**
   * The input text to use as a starting point for the edit.
   **/
  public CreateEditRequest input(String input) {
    this.input = input;
    return this;
  }

  
  @ApiModelProperty(example = "What day of the wek is it?", value = "The input text to use as a starting point for the edit.")
  @JsonProperty("input")
  public String getInput() {
    return input;
  }

  @JsonProperty("input")
  public void setInput(String input) {
    this.input = input;
  }

  /**
   * How many edits to generate for the input and instruction.
   * minimum: 1
   * maximum: 20
   **/
  public CreateEditRequest n(Integer n) {
    this.n = n;
    return this;
  }

  
  @ApiModelProperty(example = "1", value = "How many edits to generate for the input and instruction.")
  @JsonProperty("n")
 @Min(1) @Max(20)  public Integer getN() {
    return n;
  }

  @JsonProperty("n")
  public void setN(Integer n) {
    this.n = n;
  }

  /**
   * completions_temperature_description
   * minimum: 0
   * maximum: 2
   **/
  public CreateEditRequest temperature(BigDecimal temperature) {
    this.temperature = temperature;
    return this;
  }

  
  @ApiModelProperty(example = "1", value = "completions_temperature_description")
  @JsonProperty("temperature")
 @DecimalMin("0") @DecimalMax("2")  public BigDecimal getTemperature() {
    return temperature;
  }

  @JsonProperty("temperature")
  public void setTemperature(BigDecimal temperature) {
    this.temperature = temperature;
  }

  /**
   * completions_top_p_description
   * minimum: 0
   * maximum: 1
   **/
  public CreateEditRequest topP(BigDecimal topP) {
    this.topP = topP;
    return this;
  }

  
  @ApiModelProperty(example = "1", value = "completions_top_p_description")
  @JsonProperty("top_p")
 @DecimalMin("0") @DecimalMax("1")  public BigDecimal getTopP() {
    return topP;
  }

  @JsonProperty("top_p")
  public void setTopP(BigDecimal topP) {
    this.topP = topP;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateEditRequest createEditRequest = (CreateEditRequest) o;
    return Objects.equals(this.instruction, createEditRequest.instruction) &&
        Objects.equals(this.model, createEditRequest.model) &&
        Objects.equals(this.input, createEditRequest.input) &&
        Objects.equals(this.n, createEditRequest.n) &&
        Objects.equals(this.temperature, createEditRequest.temperature) &&
        Objects.equals(this.topP, createEditRequest.topP);
  }

  @Override
  public int hashCode() {
    return Objects.hash(instruction, model, input, n, temperature, topP);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateEditRequest {\n");
    
    sb.append("    instruction: ").append(toIndentedString(instruction)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    input: ").append(toIndentedString(input)).append("\n");
    sb.append("    n: ").append(toIndentedString(n)).append("\n");
    sb.append("    temperature: ").append(toIndentedString(temperature)).append("\n");
    sb.append("    topP: ").append(toIndentedString(topP)).append("\n");
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

