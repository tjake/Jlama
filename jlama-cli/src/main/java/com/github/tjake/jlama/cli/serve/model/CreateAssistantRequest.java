package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.AssistantObjectToolsInner;
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



@JsonTypeName("CreateAssistantRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateAssistantRequest   {
  private @Valid String model;
  private @Valid String name;
  private @Valid String description;
  private @Valid String instructions;
  private @Valid List<@Valid AssistantObjectToolsInner> tools = new ArrayList<>();
  private @Valid List<String> fileIds = new ArrayList<>();
  private @Valid Object metadata;

  /**
   **/
  public CreateAssistantRequest model(String model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("model")
  @NotNull
  public String getModel() {
    return model;
  }

  @JsonProperty("model")
  public void setModel(String model) {
    this.model = model;
  }

  /**
   * assistant_name_param_description
   **/
  public CreateAssistantRequest name(String name) {
    this.name = name;
    return this;
  }

  
  @ApiModelProperty(value = "assistant_name_param_description")
  @JsonProperty("name")
 @Size(max=256)  public String getName() {
    return name;
  }

  @JsonProperty("name")
  public void setName(String name) {
    this.name = name;
  }

  /**
   * assistant_description_param_description
   **/
  public CreateAssistantRequest description(String description) {
    this.description = description;
    return this;
  }

  
  @ApiModelProperty(value = "assistant_description_param_description")
  @JsonProperty("description")
 @Size(max=512)  public String getDescription() {
    return description;
  }

  @JsonProperty("description")
  public void setDescription(String description) {
    this.description = description;
  }

  /**
   * assistant_instructions_param_description
   **/
  public CreateAssistantRequest instructions(String instructions) {
    this.instructions = instructions;
    return this;
  }

  
  @ApiModelProperty(value = "assistant_instructions_param_description")
  @JsonProperty("instructions")
 @Size(max=32768)  public String getInstructions() {
    return instructions;
  }

  @JsonProperty("instructions")
  public void setInstructions(String instructions) {
    this.instructions = instructions;
  }

  /**
   * assistant_tools_param_description
   **/
  public CreateAssistantRequest tools(List<@Valid AssistantObjectToolsInner> tools) {
    this.tools = tools;
    return this;
  }

  
  @ApiModelProperty(value = "assistant_tools_param_description")
  @JsonProperty("tools")
 @Size(max=128)  public List<AssistantObjectToolsInner> getTools() {
    return tools;
  }

  @JsonProperty("tools")
  public void setTools(List<@Valid AssistantObjectToolsInner> tools) {
    this.tools = tools;
  }

  public CreateAssistantRequest addToolsItem(AssistantObjectToolsInner toolsItem) {
    if (this.tools == null) {
      this.tools = new ArrayList<>();
    }

    this.tools.add(toolsItem);
    return this;
  }

  public CreateAssistantRequest removeToolsItem(AssistantObjectToolsInner toolsItem) {
    if (toolsItem != null && this.tools != null) {
      this.tools.remove(toolsItem);
    }

    return this;
  }
  /**
   * assistant_file_param_description
   **/
  public CreateAssistantRequest fileIds(List<String> fileIds) {
    this.fileIds = fileIds;
    return this;
  }

  
  @ApiModelProperty(value = "assistant_file_param_description")
  @JsonProperty("file_ids")
 @Size(max=20)  public List<String> getFileIds() {
    return fileIds;
  }

  @JsonProperty("file_ids")
  public void setFileIds(List<String> fileIds) {
    this.fileIds = fileIds;
  }

  public CreateAssistantRequest addFileIdsItem(String fileIdsItem) {
    if (this.fileIds == null) {
      this.fileIds = new ArrayList<>();
    }

    this.fileIds.add(fileIdsItem);
    return this;
  }

  public CreateAssistantRequest removeFileIdsItem(String fileIdsItem) {
    if (fileIdsItem != null && this.fileIds != null) {
      this.fileIds.remove(fileIdsItem);
    }

    return this;
  }
  /**
   * metadata_description
   **/
  public CreateAssistantRequest metadata(Object metadata) {
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
    CreateAssistantRequest createAssistantRequest = (CreateAssistantRequest) o;
    return Objects.equals(this.model, createAssistantRequest.model) &&
        Objects.equals(this.name, createAssistantRequest.name) &&
        Objects.equals(this.description, createAssistantRequest.description) &&
        Objects.equals(this.instructions, createAssistantRequest.instructions) &&
        Objects.equals(this.tools, createAssistantRequest.tools) &&
        Objects.equals(this.fileIds, createAssistantRequest.fileIds) &&
        Objects.equals(this.metadata, createAssistantRequest.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(model, name, description, instructions, tools, fileIds, metadata);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateAssistantRequest {\n");
    
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    name: ").append(toIndentedString(name)).append("\n");
    sb.append("    description: ").append(toIndentedString(description)).append("\n");
    sb.append("    instructions: ").append(toIndentedString(instructions)).append("\n");
    sb.append("    tools: ").append(toIndentedString(tools)).append("\n");
    sb.append("    fileIds: ").append(toIndentedString(fileIds)).append("\n");
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

