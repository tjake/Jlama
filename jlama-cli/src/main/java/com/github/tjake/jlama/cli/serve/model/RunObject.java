package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.AssistantObjectToolsInner;
import com.github.tjake.jlama.cli.serve.model.RunObjectLastError;
import com.github.tjake.jlama.cli.serve.model.RunObjectRequiredAction;
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
 * Represents an execution run on a [thread](/docs/api-reference/threads).
 **/
@ApiModel(description = "Represents an execution run on a [thread](/docs/api-reference/threads).")
@JsonTypeName("RunObject")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class RunObject   {
  private @Valid String id;
  public enum ObjectEnum {

    THREAD_RUN(String.valueOf("thread.run"));


    private String value;

    ObjectEnum (String v) {
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
    public static ObjectEnum fromString(String s) {
        for (ObjectEnum b : ObjectEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected string value '" + s + "'");
    }

    @JsonCreator
    public static ObjectEnum fromValue(String value) {
        for (ObjectEnum b : ObjectEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }
}

  private @Valid ObjectEnum _object;
  private @Valid Integer createdAt;
  private @Valid String threadId;
  private @Valid String assistantId;
  public enum StatusEnum {

    QUEUED(String.valueOf("queued")), IN_PROGRESS(String.valueOf("in_progress")), REQUIRES_ACTION(String.valueOf("requires_action")), CANCELLING(String.valueOf("cancelling")), CANCELLED(String.valueOf("cancelled")), FAILED(String.valueOf("failed")), COMPLETED(String.valueOf("completed")), EXPIRED(String.valueOf("expired"));


    private String value;

    StatusEnum (String v) {
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
    public static StatusEnum fromString(String s) {
        for (StatusEnum b : StatusEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected string value '" + s + "'");
    }

    @JsonCreator
    public static StatusEnum fromValue(String value) {
        for (StatusEnum b : StatusEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }
}

  private @Valid StatusEnum status;
  private @Valid RunObjectRequiredAction requiredAction;
  private @Valid RunObjectLastError lastError;
  private @Valid Integer expiresAt;
  private @Valid Integer startedAt;
  private @Valid Integer cancelledAt;
  private @Valid Integer failedAt;
  private @Valid Integer completedAt;
  private @Valid String model;
  private @Valid String instructions;
  private @Valid List<@Valid AssistantObjectToolsInner> tools = new ArrayList<>();
  private @Valid List<String> fileIds = new ArrayList<>();
  private @Valid Object metadata;

  /**
   * The identifier, which can be referenced in API endpoints.
   **/
  public RunObject id(String id) {
    this.id = id;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The identifier, which can be referenced in API endpoints.")
  @JsonProperty("id")
  @NotNull
  public String getId() {
    return id;
  }

  @JsonProperty("id")
  public void setId(String id) {
    this.id = id;
  }

  /**
   * The object type, which is always &#x60;thread.run&#x60;.
   **/
  public RunObject _object(ObjectEnum _object) {
    this._object = _object;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The object type, which is always `thread.run`.")
  @JsonProperty("object")
  @NotNull
  public ObjectEnum getObject() {
    return _object;
  }

  @JsonProperty("object")
  public void setObject(ObjectEnum _object) {
    this._object = _object;
  }

  /**
   * The Unix timestamp (in seconds) for when the run was created.
   **/
  public RunObject createdAt(Integer createdAt) {
    this.createdAt = createdAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run was created.")
  @JsonProperty("created_at")
  @NotNull
  public Integer getCreatedAt() {
    return createdAt;
  }

  @JsonProperty("created_at")
  public void setCreatedAt(Integer createdAt) {
    this.createdAt = createdAt;
  }

  /**
   * The ID of the [thread](/docs/api-reference/threads) that was executed on as a part of this run.
   **/
  public RunObject threadId(String threadId) {
    this.threadId = threadId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The ID of the [thread](/docs/api-reference/threads) that was executed on as a part of this run.")
  @JsonProperty("thread_id")
  @NotNull
  public String getThreadId() {
    return threadId;
  }

  @JsonProperty("thread_id")
  public void setThreadId(String threadId) {
    this.threadId = threadId;
  }

  /**
   * The ID of the [assistant](/docs/api-reference/assistants) used for execution of this run.
   **/
  public RunObject assistantId(String assistantId) {
    this.assistantId = assistantId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The ID of the [assistant](/docs/api-reference/assistants) used for execution of this run.")
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
   * The status of the run, which can be either &#x60;queued&#x60;, &#x60;in_progress&#x60;, &#x60;requires_action&#x60;, &#x60;cancelling&#x60;, &#x60;cancelled&#x60;, &#x60;failed&#x60;, &#x60;completed&#x60;, or &#x60;expired&#x60;.
   **/
  public RunObject status(StatusEnum status) {
    this.status = status;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The status of the run, which can be either `queued`, `in_progress`, `requires_action`, `cancelling`, `cancelled`, `failed`, `completed`, or `expired`.")
  @JsonProperty("status")
  @NotNull
  public StatusEnum getStatus() {
    return status;
  }

  @JsonProperty("status")
  public void setStatus(StatusEnum status) {
    this.status = status;
  }

  /**
   **/
  public RunObject requiredAction(RunObjectRequiredAction requiredAction) {
    this.requiredAction = requiredAction;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("required_action")
  @NotNull
  public RunObjectRequiredAction getRequiredAction() {
    return requiredAction;
  }

  @JsonProperty("required_action")
  public void setRequiredAction(RunObjectRequiredAction requiredAction) {
    this.requiredAction = requiredAction;
  }

  /**
   **/
  public RunObject lastError(RunObjectLastError lastError) {
    this.lastError = lastError;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("last_error")
  @NotNull
  public RunObjectLastError getLastError() {
    return lastError;
  }

  @JsonProperty("last_error")
  public void setLastError(RunObjectLastError lastError) {
    this.lastError = lastError;
  }

  /**
   * The Unix timestamp (in seconds) for when the run will expire.
   **/
  public RunObject expiresAt(Integer expiresAt) {
    this.expiresAt = expiresAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run will expire.")
  @JsonProperty("expires_at")
  @NotNull
  public Integer getExpiresAt() {
    return expiresAt;
  }

  @JsonProperty("expires_at")
  public void setExpiresAt(Integer expiresAt) {
    this.expiresAt = expiresAt;
  }

  /**
   * The Unix timestamp (in seconds) for when the run was started.
   **/
  public RunObject startedAt(Integer startedAt) {
    this.startedAt = startedAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run was started.")
  @JsonProperty("started_at")
  @NotNull
  public Integer getStartedAt() {
    return startedAt;
  }

  @JsonProperty("started_at")
  public void setStartedAt(Integer startedAt) {
    this.startedAt = startedAt;
  }

  /**
   * The Unix timestamp (in seconds) for when the run was cancelled.
   **/
  public RunObject cancelledAt(Integer cancelledAt) {
    this.cancelledAt = cancelledAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run was cancelled.")
  @JsonProperty("cancelled_at")
  @NotNull
  public Integer getCancelledAt() {
    return cancelledAt;
  }

  @JsonProperty("cancelled_at")
  public void setCancelledAt(Integer cancelledAt) {
    this.cancelledAt = cancelledAt;
  }

  /**
   * The Unix timestamp (in seconds) for when the run failed.
   **/
  public RunObject failedAt(Integer failedAt) {
    this.failedAt = failedAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run failed.")
  @JsonProperty("failed_at")
  @NotNull
  public Integer getFailedAt() {
    return failedAt;
  }

  @JsonProperty("failed_at")
  public void setFailedAt(Integer failedAt) {
    this.failedAt = failedAt;
  }

  /**
   * The Unix timestamp (in seconds) for when the run was completed.
   **/
  public RunObject completedAt(Integer completedAt) {
    this.completedAt = completedAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run was completed.")
  @JsonProperty("completed_at")
  @NotNull
  public Integer getCompletedAt() {
    return completedAt;
  }

  @JsonProperty("completed_at")
  public void setCompletedAt(Integer completedAt) {
    this.completedAt = completedAt;
  }

  /**
   * The model that the [assistant](/docs/api-reference/assistants) used for this run.
   **/
  public RunObject model(String model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The model that the [assistant](/docs/api-reference/assistants) used for this run.")
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
   * The instructions that the [assistant](/docs/api-reference/assistants) used for this run.
   **/
  public RunObject instructions(String instructions) {
    this.instructions = instructions;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The instructions that the [assistant](/docs/api-reference/assistants) used for this run.")
  @JsonProperty("instructions")
  @NotNull
  public String getInstructions() {
    return instructions;
  }

  @JsonProperty("instructions")
  public void setInstructions(String instructions) {
    this.instructions = instructions;
  }

  /**
   * The list of tools that the [assistant](/docs/api-reference/assistants) used for this run.
   **/
  public RunObject tools(List<@Valid AssistantObjectToolsInner> tools) {
    this.tools = tools;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The list of tools that the [assistant](/docs/api-reference/assistants) used for this run.")
  @JsonProperty("tools")
  @NotNull
 @Size(max=20)  public List<AssistantObjectToolsInner> getTools() {
    return tools;
  }

  @JsonProperty("tools")
  public void setTools(List<@Valid AssistantObjectToolsInner> tools) {
    this.tools = tools;
  }

  public RunObject addToolsItem(AssistantObjectToolsInner toolsItem) {
    if (this.tools == null) {
      this.tools = new ArrayList<>();
    }

    this.tools.add(toolsItem);
    return this;
  }

  public RunObject removeToolsItem(AssistantObjectToolsInner toolsItem) {
    if (toolsItem != null && this.tools != null) {
      this.tools.remove(toolsItem);
    }

    return this;
  }
  /**
   * The list of [File](/docs/api-reference/files) IDs the [assistant](/docs/api-reference/assistants) used for this run.
   **/
  public RunObject fileIds(List<String> fileIds) {
    this.fileIds = fileIds;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The list of [File](/docs/api-reference/files) IDs the [assistant](/docs/api-reference/assistants) used for this run.")
  @JsonProperty("file_ids")
  @NotNull
  public List<String> getFileIds() {
    return fileIds;
  }

  @JsonProperty("file_ids")
  public void setFileIds(List<String> fileIds) {
    this.fileIds = fileIds;
  }

  public RunObject addFileIdsItem(String fileIdsItem) {
    if (this.fileIds == null) {
      this.fileIds = new ArrayList<>();
    }

    this.fileIds.add(fileIdsItem);
    return this;
  }

  public RunObject removeFileIdsItem(String fileIdsItem) {
    if (fileIdsItem != null && this.fileIds != null) {
      this.fileIds.remove(fileIdsItem);
    }

    return this;
  }
  /**
   * metadata_description
   **/
  public RunObject metadata(Object metadata) {
    this.metadata = metadata;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "metadata_description")
  @JsonProperty("metadata")
  @NotNull
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
    RunObject runObject = (RunObject) o;
    return Objects.equals(this.id, runObject.id) &&
        Objects.equals(this._object, runObject._object) &&
        Objects.equals(this.createdAt, runObject.createdAt) &&
        Objects.equals(this.threadId, runObject.threadId) &&
        Objects.equals(this.assistantId, runObject.assistantId) &&
        Objects.equals(this.status, runObject.status) &&
        Objects.equals(this.requiredAction, runObject.requiredAction) &&
        Objects.equals(this.lastError, runObject.lastError) &&
        Objects.equals(this.expiresAt, runObject.expiresAt) &&
        Objects.equals(this.startedAt, runObject.startedAt) &&
        Objects.equals(this.cancelledAt, runObject.cancelledAt) &&
        Objects.equals(this.failedAt, runObject.failedAt) &&
        Objects.equals(this.completedAt, runObject.completedAt) &&
        Objects.equals(this.model, runObject.model) &&
        Objects.equals(this.instructions, runObject.instructions) &&
        Objects.equals(this.tools, runObject.tools) &&
        Objects.equals(this.fileIds, runObject.fileIds) &&
        Objects.equals(this.metadata, runObject.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, _object, createdAt, threadId, assistantId, status, requiredAction, lastError, expiresAt, startedAt, cancelledAt, failedAt, completedAt, model, instructions, tools, fileIds, metadata);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class RunObject {\n");
    
    sb.append("    id: ").append(toIndentedString(id)).append("\n");
    sb.append("    _object: ").append(toIndentedString(_object)).append("\n");
    sb.append("    createdAt: ").append(toIndentedString(createdAt)).append("\n");
    sb.append("    threadId: ").append(toIndentedString(threadId)).append("\n");
    sb.append("    assistantId: ").append(toIndentedString(assistantId)).append("\n");
    sb.append("    status: ").append(toIndentedString(status)).append("\n");
    sb.append("    requiredAction: ").append(toIndentedString(requiredAction)).append("\n");
    sb.append("    lastError: ").append(toIndentedString(lastError)).append("\n");
    sb.append("    expiresAt: ").append(toIndentedString(expiresAt)).append("\n");
    sb.append("    startedAt: ").append(toIndentedString(startedAt)).append("\n");
    sb.append("    cancelledAt: ").append(toIndentedString(cancelledAt)).append("\n");
    sb.append("    failedAt: ").append(toIndentedString(failedAt)).append("\n");
    sb.append("    completedAt: ").append(toIndentedString(completedAt)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
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

