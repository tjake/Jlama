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



@JsonTypeName("ModifyAssistantRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class ModifyAssistantRequest   {
  private @Valid String model;
  private @Valid String name;
  private @Valid String description;
  private @Valid String instructions;
  private @Valid List<@Valid AssistantObjectToolsInner> tools = new ArrayList<>();
  private @Valid List<String> fileIds = new ArrayList<>();
  private @Valid Object metadata;

  /**
   **/
  public ModifyAssistantRequest model(String model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(value = "")
  @JsonProperty("model")
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
  public ModifyAssistantRequest name(String name) {
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
  public ModifyAssistantRequest description(String description) {
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
  public ModifyAssistantRequest instructions(String instructions) {
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
  public ModifyAssistantRequest tools(List<@Valid AssistantObjectToolsInner> tools) {
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

  public ModifyAssistantRequest addToolsItem(AssistantObjectToolsInner toolsItem) {
    if (this.tools == null) {
      this.tools = new ArrayList<>();
    }

    this.tools.add(toolsItem);
    return this;
  }

  public ModifyAssistantRequest removeToolsItem(AssistantObjectToolsInner toolsItem) {
    if (toolsItem != null && this.tools != null) {
      this.tools.remove(toolsItem);
    }

    return this;
  }
  /**
   * A list of [File](/docs/api-reference/files) IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order. If a file was previosuly attached to the list but does not show up in the list, it will be deleted from the assistant. 
   **/
  public ModifyAssistantRequest fileIds(List<String> fileIds) {
    this.fileIds = fileIds;
    return this;
  }

  
  @ApiModelProperty(value = "A list of [File](/docs/api-reference/files) IDs attached to this assistant. There can be a maximum of 20 files attached to the assistant. Files are ordered by their creation date in ascending order. If a file was previosuly attached to the list but does not show up in the list, it will be deleted from the assistant. ")
  @JsonProperty("file_ids")
 @Size(max=20)  public List<String> getFileIds() {
    return fileIds;
  }

  @JsonProperty("file_ids")
  public void setFileIds(List<String> fileIds) {
    this.fileIds = fileIds;
  }

  public ModifyAssistantRequest addFileIdsItem(String fileIdsItem) {
    if (this.fileIds == null) {
      this.fileIds = new ArrayList<>();
    }

    this.fileIds.add(fileIdsItem);
    return this;
  }

  public ModifyAssistantRequest removeFileIdsItem(String fileIdsItem) {
    if (fileIdsItem != null && this.fileIds != null) {
      this.fileIds.remove(fileIdsItem);
    }

    return this;
  }
  /**
   * metadata_description
   **/
  public ModifyAssistantRequest metadata(Object metadata) {
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
    ModifyAssistantRequest modifyAssistantRequest = (ModifyAssistantRequest) o;
    return Objects.equals(this.model, modifyAssistantRequest.model) &&
        Objects.equals(this.name, modifyAssistantRequest.name) &&
        Objects.equals(this.description, modifyAssistantRequest.description) &&
        Objects.equals(this.instructions, modifyAssistantRequest.instructions) &&
        Objects.equals(this.tools, modifyAssistantRequest.tools) &&
        Objects.equals(this.fileIds, modifyAssistantRequest.fileIds) &&
        Objects.equals(this.metadata, modifyAssistantRequest.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(model, name, description, instructions, tools, fileIds, metadata);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class ModifyAssistantRequest {\n");
    
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

