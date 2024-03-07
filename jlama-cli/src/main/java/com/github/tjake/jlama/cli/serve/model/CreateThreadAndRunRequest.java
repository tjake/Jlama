package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CreateThreadAndRunRequestToolsInner;
import com.github.tjake.jlama.cli.serve.model.CreateThreadRequest;
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



@JsonTypeName("CreateThreadAndRunRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateThreadAndRunRequest   {
  private @Valid String assistantId;
  private @Valid CreateThreadRequest thread;
  private @Valid String model;
  private @Valid String instructions;
  private @Valid List<@Valid CreateThreadAndRunRequestToolsInner> tools;
  private @Valid Object metadata;

  /**
   * The ID of the [assistant](/docs/api-reference/assistants) to use to execute this run.
   **/
  public CreateThreadAndRunRequest assistantId(String assistantId) {
    this.assistantId = assistantId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The ID of the [assistant](/docs/api-reference/assistants) to use to execute this run.")
  @JsonProperty("assistant_id")
  @NotNull
  public String getAssistantId() {
    return assistantId;
  }

  @JsonProperty("assistant_id")
  public void setAssistantId(String assistantId) {
    this.assistantId = assistantId;
  }

  /**
   **/
  public CreateThreadAndRunRequest thread(CreateThreadRequest thread) {
    this.thread = thread;
    return this;
  }

  
  @ApiModelProperty(value = "")
  @JsonProperty("thread")
  public CreateThreadRequest getThread() {
    return thread;
  }

  @JsonProperty("thread")
  public void setThread(CreateThreadRequest thread) {
    this.thread = thread;
  }

  /**
   * The ID of the [Model](/docs/api-reference/models) to be used to execute this run. If a value is provided here, it will override the model associated with the assistant. If not, the model associated with the assistant will be used.
   **/
  public CreateThreadAndRunRequest model(String model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(value = "The ID of the [Model](/docs/api-reference/models) to be used to execute this run. If a value is provided here, it will override the model associated with the assistant. If not, the model associated with the assistant will be used.")
  @JsonProperty("model")
  public String getModel() {
    return model;
  }

  @JsonProperty("model")
  public void setModel(String model) {
    this.model = model;
  }

  /**
   * Override the default system message of the assistant. This is useful for modifying the behavior on a per-run basis.
   **/
  public CreateThreadAndRunRequest instructions(String instructions) {
    this.instructions = instructions;
    return this;
  }

  
  @ApiModelProperty(value = "Override the default system message of the assistant. This is useful for modifying the behavior on a per-run basis.")
  @JsonProperty("instructions")
  public String getInstructions() {
    return instructions;
  }

  @JsonProperty("instructions")
  public void setInstructions(String instructions) {
    this.instructions = instructions;
  }

  /**
   * Override the tools the assistant can use for this run. This is useful for modifying the behavior on a per-run basis.
   **/
  public CreateThreadAndRunRequest tools(List<@Valid CreateThreadAndRunRequestToolsInner> tools) {
    this.tools = tools;
    return this;
  }

  
  @ApiModelProperty(value = "Override the tools the assistant can use for this run. This is useful for modifying the behavior on a per-run basis.")
  @JsonProperty("tools")
 @Size(max=20)  public List<CreateThreadAndRunRequestToolsInner> getTools() {
    return tools;
  }

  @JsonProperty("tools")
  public void setTools(List<@Valid CreateThreadAndRunRequestToolsInner> tools) {
    this.tools = tools;
  }

  public CreateThreadAndRunRequest addToolsItem(CreateThreadAndRunRequestToolsInner toolsItem) {
    if (this.tools == null) {
      this.tools = new ArrayList<>();
    }

    this.tools.add(toolsItem);
    return this;
  }

  public CreateThreadAndRunRequest removeToolsItem(CreateThreadAndRunRequestToolsInner toolsItem) {
    if (toolsItem != null && this.tools != null) {
      this.tools.remove(toolsItem);
    }

    return this;
  }
  /**
   * metadata_description
   **/
  public CreateThreadAndRunRequest metadata(Object metadata) {
    this.metadata = metadata;
    return this;
  }

  
  @ApiModelProperty(value = "metadata_description")
  @JsonProperty("metadata")
  public Object getMetadata() {
    return metadata;
  }

  @JsonProperty("metadata")
  public void setMetadata(Object metadata) {
    this.metadata = metadata;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateThreadAndRunRequest createThreadAndRunRequest = (CreateThreadAndRunRequest) o;
    return Objects.equals(this.assistantId, createThreadAndRunRequest.assistantId) &&
        Objects.equals(this.thread, createThreadAndRunRequest.thread) &&
        Objects.equals(this.model, createThreadAndRunRequest.model) &&
        Objects.equals(this.instructions, createThreadAndRunRequest.instructions) &&
        Objects.equals(this.tools, createThreadAndRunRequest.tools) &&
        Objects.equals(this.metadata, createThreadAndRunRequest.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(assistantId, thread, model, instructions, tools, metadata);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateThreadAndRunRequest {\n");
    
    sb.append("    assistantId: ").append(toIndentedString(assistantId)).append("\n");
    sb.append("    thread: ").append(toIndentedString(thread)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    instructions: ").append(toIndentedString(instructions)).append("\n");
    sb.append("    tools: ").append(toIndentedString(tools)).append("\n");
    sb.append("    metadata: ").append(toIndentedString(metadata)).append("\n");
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

