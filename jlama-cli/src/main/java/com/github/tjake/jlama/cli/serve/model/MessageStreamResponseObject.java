package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.MessageContentDeltaObject;
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
 * Represents a chunk or part of a message in a streaming context, including incremental updates.
 **/
@ApiModel(description = "Represents a chunk or part of a message in a streaming context, including incremental updates.")
@JsonTypeName("MessageStreamResponseObject")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class MessageStreamResponseObject   {
  private @Valid String id;
  public enum ObjectEnum {

    THREAD_MESSAGE_STREAM_PART(String.valueOf("thread.message.stream.part"));


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
  public enum RoleEnum {

    USER(String.valueOf("user")), ASSISTANT(String.valueOf("assistant"));


    private String value;

    RoleEnum (String v) {
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
    public static RoleEnum fromString(String s) {
        for (RoleEnum b : RoleEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected string value '" + s + "'");
    }

    @JsonCreator
    public static RoleEnum fromValue(String value) {
        for (RoleEnum b : RoleEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }
}

  private @Valid RoleEnum role;
  private @Valid List<@Valid MessageContentDeltaObject> content = new ArrayList<>();
  private @Valid String assistantId;
  private @Valid String runId;
  private @Valid List<String> fileIds = new ArrayList<>();
  private @Valid Object metadata;

  /**
   * A unique identifier for this part of the streamed message.
   **/
  public MessageStreamResponseObject id(String id) {
    this.id = id;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "A unique identifier for this part of the streamed message.")
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
   * The object type, which is always &#x60;thread.message.stream.part&#x60;.
   **/
  public MessageStreamResponseObject _object(ObjectEnum _object) {
    this._object = _object;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The object type, which is always `thread.message.stream.part`.")
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
   * The Unix timestamp (in seconds) for when this part of the message was created.
   **/
  public MessageStreamResponseObject createdAt(Integer createdAt) {
    this.createdAt = createdAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when this part of the message was created.")
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
   * The thread ID that this message part belongs to.
   **/
  public MessageStreamResponseObject threadId(String threadId) {
    this.threadId = threadId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The thread ID that this message part belongs to.")
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
   * The entity that produced this part of the message. One of &#x60;user&#x60; or &#x60;assistant&#x60;.
   **/
  public MessageStreamResponseObject role(RoleEnum role) {
    this.role = role;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The entity that produced this part of the message. One of `user` or `assistant`.")
  @JsonProperty("role")
  @NotNull
  public RoleEnum getRole() {
    return role;
  }

  @JsonProperty("role")
  public void setRole(RoleEnum role) {
    this.role = role;
  }

  /**
   * An array of incremental updates for this part of the message, each represented by a delta.
   **/
  public MessageStreamResponseObject content(List<@Valid MessageContentDeltaObject> content) {
    this.content = content;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "An array of incremental updates for this part of the message, each represented by a delta.")
  @JsonProperty("content")
  @NotNull
  public List<MessageContentDeltaObject> getContent() {
    return content;
  }

  @JsonProperty("content")
  public void setContent(List<@Valid MessageContentDeltaObject> content) {
    this.content = content;
  }

  public MessageStreamResponseObject addContentItem(MessageContentDeltaObject contentItem) {
    if (this.content == null) {
      this.content = new ArrayList<>();
    }

    this.content.add(contentItem);
    return this;
  }

  public MessageStreamResponseObject removeContentItem(MessageContentDeltaObject contentItem) {
    if (contentItem != null && this.content != null) {
      this.content.remove(contentItem);
    }

    return this;
  }
  /**
   * If applicable, the ID of the [assistant](/docs/api-reference/assistants) that authored this message.
   **/
  public MessageStreamResponseObject assistantId(String assistantId) {
    this.assistantId = assistantId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "If applicable, the ID of the [assistant](/docs/api-reference/assistants) that authored this message.")
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
   * If applicable, the ID of the [run](/docs/api-reference/runs) associated with the authoring of this message.
   **/
  public MessageStreamResponseObject runId(String runId) {
    this.runId = runId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "If applicable, the ID of the [run](/docs/api-reference/runs) associated with the authoring of this message.")
  @JsonProperty("run_id")
  @NotNull
  public String getRunId() {
    return runId;
  }

  @JsonProperty("run_id")
  public void setRunId(String runId) {
    this.runId = runId;
  }

  /**
   * A list of [file](/docs/api-reference/files) IDs that the assistant should use. Useful for tools like retrieval and code_interpreter that can access files. A maximum of 10 files can be attached to a message.
   **/
  public MessageStreamResponseObject fileIds(List<String> fileIds) {
    this.fileIds = fileIds;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "A list of [file](/docs/api-reference/files) IDs that the assistant should use. Useful for tools like retrieval and code_interpreter that can access files. A maximum of 10 files can be attached to a message.")
  @JsonProperty("file_ids")
  @NotNull
 @Size(max=10)  public List<String> getFileIds() {
    return fileIds;
  }

  @JsonProperty("file_ids")
  public void setFileIds(List<String> fileIds) {
    this.fileIds = fileIds;
  }

  public MessageStreamResponseObject addFileIdsItem(String fileIdsItem) {
    if (this.fileIds == null) {
      this.fileIds = new ArrayList<>();
    }

    this.fileIds.add(fileIdsItem);
    return this;
  }

  public MessageStreamResponseObject removeFileIdsItem(String fileIdsItem) {
    if (fileIdsItem != null && this.fileIds != null) {
      this.fileIds.remove(fileIdsItem);
    }

    return this;
  }
  /**
   * metadata_description
   **/
  public MessageStreamResponseObject metadata(Object metadata) {
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
    MessageStreamResponseObject messageStreamResponseObject = (MessageStreamResponseObject) o;
    return Objects.equals(this.id, messageStreamResponseObject.id) &&
        Objects.equals(this._object, messageStreamResponseObject._object) &&
        Objects.equals(this.createdAt, messageStreamResponseObject.createdAt) &&
        Objects.equals(this.threadId, messageStreamResponseObject.threadId) &&
        Objects.equals(this.role, messageStreamResponseObject.role) &&
        Objects.equals(this.content, messageStreamResponseObject.content) &&
        Objects.equals(this.assistantId, messageStreamResponseObject.assistantId) &&
        Objects.equals(this.runId, messageStreamResponseObject.runId) &&
        Objects.equals(this.fileIds, messageStreamResponseObject.fileIds) &&
        Objects.equals(this.metadata, messageStreamResponseObject.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, _object, createdAt, threadId, role, content, assistantId, runId, fileIds, metadata);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class MessageStreamResponseObject {\n");
    
    sb.append("    id: ").append(toIndentedString(id)).append("\n");
    sb.append("    _object: ").append(toIndentedString(_object)).append("\n");
    sb.append("    createdAt: ").append(toIndentedString(createdAt)).append("\n");
    sb.append("    threadId: ").append(toIndentedString(threadId)).append("\n");
    sb.append("    role: ").append(toIndentedString(role)).append("\n");
    sb.append("    content: ").append(toIndentedString(content)).append("\n");
    sb.append("    assistantId: ").append(toIndentedString(assistantId)).append("\n");
    sb.append("    runId: ").append(toIndentedString(runId)).append("\n");
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

