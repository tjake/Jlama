package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.RunStepObjectLastError;
import com.github.tjake.jlama.cli.serve.model.RunStepObjectStepDetails;
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
 * Represents a step in execution of a run. 
 **/
@ApiModel(description = "Represents a step in execution of a run. ")
@JsonTypeName("RunStepObject")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class RunStepObject   {
  private @Valid String id;
  public enum ObjectEnum {

    THREAD_RUN_STEP(String.valueOf("thread.run.step"));


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
  private @Valid String assistantId;
  private @Valid String threadId;
  private @Valid String runId;
  public enum TypeEnum {

    MESSAGE_CREATION(String.valueOf("message_creation")), TOOL_CALLS(String.valueOf("tool_calls"));


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
  public enum StatusEnum {

    IN_PROGRESS(String.valueOf("in_progress")), CANCELLED(String.valueOf("cancelled")), FAILED(String.valueOf("failed")), COMPLETED(String.valueOf("completed")), EXPIRED(String.valueOf("expired"));


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
  private @Valid RunStepObjectStepDetails stepDetails;
  private @Valid RunStepObjectLastError lastError;
  private @Valid Integer expiredAt;
  private @Valid Integer cancelledAt;
  private @Valid Integer failedAt;
  private @Valid Integer completedAt;
  private @Valid Object metadata;

  /**
   * The identifier of the run step, which can be referenced in API endpoints.
   **/
  public RunStepObject id(String id) {
    this.id = id;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The identifier of the run step, which can be referenced in API endpoints.")
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
   * The object type, which is always &#x60;thread.run.step&#x60;&#x60;.
   **/
  public RunStepObject _object(ObjectEnum _object) {
    this._object = _object;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The object type, which is always `thread.run.step``.")
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
   * The Unix timestamp (in seconds) for when the run step was created.
   **/
  public RunStepObject createdAt(Integer createdAt) {
    this.createdAt = createdAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run step was created.")
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
   * The ID of the [assistant](/docs/api-reference/assistants) associated with the run step.
   **/
  public RunStepObject assistantId(String assistantId) {
    this.assistantId = assistantId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The ID of the [assistant](/docs/api-reference/assistants) associated with the run step.")
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
   * The ID of the [thread](/docs/api-reference/threads) that was run.
   **/
  public RunStepObject threadId(String threadId) {
    this.threadId = threadId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The ID of the [thread](/docs/api-reference/threads) that was run.")
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
   * The ID of the [run](/docs/api-reference/runs) that this run step is a part of.
   **/
  public RunStepObject runId(String runId) {
    this.runId = runId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The ID of the [run](/docs/api-reference/runs) that this run step is a part of.")
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
   * The type of run step, which can be either &#x60;message_creation&#x60; or &#x60;tool_calls&#x60;.
   **/
  public RunStepObject type(TypeEnum type) {
    this.type = type;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The type of run step, which can be either `message_creation` or `tool_calls`.")
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
   * The status of the run step, which can be either &#x60;in_progress&#x60;, &#x60;cancelled&#x60;, &#x60;failed&#x60;, &#x60;completed&#x60;, or &#x60;expired&#x60;.
   **/
  public RunStepObject status(StatusEnum status) {
    this.status = status;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The status of the run step, which can be either `in_progress`, `cancelled`, `failed`, `completed`, or `expired`.")
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
  public RunStepObject stepDetails(RunStepObjectStepDetails stepDetails) {
    this.stepDetails = stepDetails;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("step_details")
  @NotNull
  public RunStepObjectStepDetails getStepDetails() {
    return stepDetails;
  }

  @JsonProperty("step_details")
  public void setStepDetails(RunStepObjectStepDetails stepDetails) {
    this.stepDetails = stepDetails;
  }

  /**
   **/
  public RunStepObject lastError(RunStepObjectLastError lastError) {
    this.lastError = lastError;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("last_error")
  @NotNull
  public RunStepObjectLastError getLastError() {
    return lastError;
  }

  @JsonProperty("last_error")
  public void setLastError(RunStepObjectLastError lastError) {
    this.lastError = lastError;
  }

  /**
   * The Unix timestamp (in seconds) for when the run step expired. A step is considered expired if the parent run is expired.
   **/
  public RunStepObject expiredAt(Integer expiredAt) {
    this.expiredAt = expiredAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run step expired. A step is considered expired if the parent run is expired.")
  @JsonProperty("expired_at")
  @NotNull
  public Integer getExpiredAt() {
    return expiredAt;
  }

  @JsonProperty("expired_at")
  public void setExpiredAt(Integer expiredAt) {
    this.expiredAt = expiredAt;
  }

  /**
   * The Unix timestamp (in seconds) for when the run step was cancelled.
   **/
  public RunStepObject cancelledAt(Integer cancelledAt) {
    this.cancelledAt = cancelledAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run step was cancelled.")
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
   * The Unix timestamp (in seconds) for when the run step failed.
   **/
  public RunStepObject failedAt(Integer failedAt) {
    this.failedAt = failedAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run step failed.")
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
   * The Unix timestamp (in seconds) for when the run step completed.
   **/
  public RunStepObject completedAt(Integer completedAt) {
    this.completedAt = completedAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the run step completed.")
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
   * metadata_description
   **/
  public RunStepObject metadata(Object metadata) {
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
    RunStepObject runStepObject = (RunStepObject) o;
    return Objects.equals(this.id, runStepObject.id) &&
        Objects.equals(this._object, runStepObject._object) &&
        Objects.equals(this.createdAt, runStepObject.createdAt) &&
        Objects.equals(this.assistantId, runStepObject.assistantId) &&
        Objects.equals(this.threadId, runStepObject.threadId) &&
        Objects.equals(this.runId, runStepObject.runId) &&
        Objects.equals(this.type, runStepObject.type) &&
        Objects.equals(this.status, runStepObject.status) &&
        Objects.equals(this.stepDetails, runStepObject.stepDetails) &&
        Objects.equals(this.lastError, runStepObject.lastError) &&
        Objects.equals(this.expiredAt, runStepObject.expiredAt) &&
        Objects.equals(this.cancelledAt, runStepObject.cancelledAt) &&
        Objects.equals(this.failedAt, runStepObject.failedAt) &&
        Objects.equals(this.completedAt, runStepObject.completedAt) &&
        Objects.equals(this.metadata, runStepObject.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, _object, createdAt, assistantId, threadId, runId, type, status, stepDetails, lastError, expiredAt, cancelledAt, failedAt, completedAt, metadata);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class RunStepObject {\n");
    
    sb.append("    id: ").append(toIndentedString(id)).append("\n");
    sb.append("    _object: ").append(toIndentedString(_object)).append("\n");
    sb.append("    createdAt: ").append(toIndentedString(createdAt)).append("\n");
    sb.append("    assistantId: ").append(toIndentedString(assistantId)).append("\n");
    sb.append("    threadId: ").append(toIndentedString(threadId)).append("\n");
    sb.append("    runId: ").append(toIndentedString(runId)).append("\n");
    sb.append("    type: ").append(toIndentedString(type)).append("\n");
    sb.append("    status: ").append(toIndentedString(status)).append("\n");
    sb.append("    stepDetails: ").append(toIndentedString(stepDetails)).append("\n");
    sb.append("    lastError: ").append(toIndentedString(lastError)).append("\n");
    sb.append("    expiredAt: ").append(toIndentedString(expiredAt)).append("\n");
    sb.append("    cancelledAt: ").append(toIndentedString(cancelledAt)).append("\n");
    sb.append("    failedAt: ").append(toIndentedString(failedAt)).append("\n");
    sb.append("    completedAt: ").append(toIndentedString(completedAt)).append("\n");
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

