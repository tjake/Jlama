package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.FineTuneEvent;
import com.github.tjake.jlama.cli.serve.model.FineTuneHyperparams;
import com.github.tjake.jlama.cli.serve.model.OpenAIFile;
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
 * The &#x60;FineTune&#x60; object represents a legacy fine-tune job that has been created through the API. 
 **/
@ApiModel(description = "The `FineTune` object represents a legacy fine-tune job that has been created through the API. ")
@JsonTypeName("FineTune")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class FineTune   {
  private @Valid String id;
  private @Valid Integer createdAt;
  private @Valid List<@Valid FineTuneEvent> events;
  private @Valid String fineTunedModel;
  private @Valid FineTuneHyperparams hyperparams;
  private @Valid String model;
  public enum ObjectEnum {

    FINE_TUNE(String.valueOf("fine-tune"));


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
  private @Valid String organizationId;
  private @Valid List<@Valid OpenAIFile> resultFiles = new ArrayList<>();
  private @Valid String status;
  private @Valid List<@Valid OpenAIFile> trainingFiles = new ArrayList<>();
  private @Valid Integer updatedAt;
  private @Valid List<@Valid OpenAIFile> validationFiles = new ArrayList<>();

  /**
   * The object identifier, which can be referenced in the API endpoints.
   **/
  public FineTune id(String id) {
    this.id = id;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The object identifier, which can be referenced in the API endpoints.")
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
   * The Unix timestamp (in seconds) for when the fine-tuning job was created.
   **/
  public FineTune createdAt(Integer createdAt) {
    this.createdAt = createdAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the fine-tuning job was created.")
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
   * The list of events that have been observed in the lifecycle of the FineTune job.
   **/
  public FineTune events(List<@Valid FineTuneEvent> events) {
    this.events = events;
    return this;
  }

  
  @ApiModelProperty(value = "The list of events that have been observed in the lifecycle of the FineTune job.")
  @JsonProperty("events")
  public List<FineTuneEvent> getEvents() {
    return events;
  }

  @JsonProperty("events")
  public void setEvents(List<@Valid FineTuneEvent> events) {
    this.events = events;
  }

  public FineTune addEventsItem(FineTuneEvent eventsItem) {
    if (this.events == null) {
      this.events = new ArrayList<>();
    }

    this.events.add(eventsItem);
    return this;
  }

  public FineTune removeEventsItem(FineTuneEvent eventsItem) {
    if (eventsItem != null && this.events != null) {
      this.events.remove(eventsItem);
    }

    return this;
  }
  /**
   * The name of the fine-tuned model that is being created.
   **/
  public FineTune fineTunedModel(String fineTunedModel) {
    this.fineTunedModel = fineTunedModel;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The name of the fine-tuned model that is being created.")
  @JsonProperty("fine_tuned_model")
  @NotNull
  public String getFineTunedModel() {
    return fineTunedModel;
  }

  @JsonProperty("fine_tuned_model")
  public void setFineTunedModel(String fineTunedModel) {
    this.fineTunedModel = fineTunedModel;
  }

  /**
   **/
  public FineTune hyperparams(FineTuneHyperparams hyperparams) {
    this.hyperparams = hyperparams;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("hyperparams")
  @NotNull
  public FineTuneHyperparams getHyperparams() {
    return hyperparams;
  }

  @JsonProperty("hyperparams")
  public void setHyperparams(FineTuneHyperparams hyperparams) {
    this.hyperparams = hyperparams;
  }

  /**
   * The base model that is being fine-tuned.
   **/
  public FineTune model(String model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The base model that is being fine-tuned.")
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
   * The object type, which is always \&quot;fine-tune\&quot;.
   **/
  public FineTune _object(ObjectEnum _object) {
    this._object = _object;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The object type, which is always \"fine-tune\".")
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
   * The organization that owns the fine-tuning job.
   **/
  public FineTune organizationId(String organizationId) {
    this.organizationId = organizationId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The organization that owns the fine-tuning job.")
  @JsonProperty("organization_id")
  @NotNull
  public String getOrganizationId() {
    return organizationId;
  }

  @JsonProperty("organization_id")
  public void setOrganizationId(String organizationId) {
    this.organizationId = organizationId;
  }

  /**
   * The compiled results files for the fine-tuning job.
   **/
  public FineTune resultFiles(List<@Valid OpenAIFile> resultFiles) {
    this.resultFiles = resultFiles;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The compiled results files for the fine-tuning job.")
  @JsonProperty("result_files")
  @NotNull
  public List<OpenAIFile> getResultFiles() {
    return resultFiles;
  }

  @JsonProperty("result_files")
  public void setResultFiles(List<@Valid OpenAIFile> resultFiles) {
    this.resultFiles = resultFiles;
  }

  public FineTune addResultFilesItem(OpenAIFile resultFilesItem) {
    if (this.resultFiles == null) {
      this.resultFiles = new ArrayList<>();
    }

    this.resultFiles.add(resultFilesItem);
    return this;
  }

  public FineTune removeResultFilesItem(OpenAIFile resultFilesItem) {
    if (resultFilesItem != null && this.resultFiles != null) {
      this.resultFiles.remove(resultFilesItem);
    }

    return this;
  }
  /**
   * The current status of the fine-tuning job, which can be either &#x60;created&#x60;, &#x60;running&#x60;, &#x60;succeeded&#x60;, &#x60;failed&#x60;, or &#x60;cancelled&#x60;.
   **/
  public FineTune status(String status) {
    this.status = status;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The current status of the fine-tuning job, which can be either `created`, `running`, `succeeded`, `failed`, or `cancelled`.")
  @JsonProperty("status")
  @NotNull
  public String getStatus() {
    return status;
  }

  @JsonProperty("status")
  public void setStatus(String status) {
    this.status = status;
  }

  /**
   * The list of files used for training.
   **/
  public FineTune trainingFiles(List<@Valid OpenAIFile> trainingFiles) {
    this.trainingFiles = trainingFiles;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The list of files used for training.")
  @JsonProperty("training_files")
  @NotNull
  public List<OpenAIFile> getTrainingFiles() {
    return trainingFiles;
  }

  @JsonProperty("training_files")
  public void setTrainingFiles(List<@Valid OpenAIFile> trainingFiles) {
    this.trainingFiles = trainingFiles;
  }

  public FineTune addTrainingFilesItem(OpenAIFile trainingFilesItem) {
    if (this.trainingFiles == null) {
      this.trainingFiles = new ArrayList<>();
    }

    this.trainingFiles.add(trainingFilesItem);
    return this;
  }

  public FineTune removeTrainingFilesItem(OpenAIFile trainingFilesItem) {
    if (trainingFilesItem != null && this.trainingFiles != null) {
      this.trainingFiles.remove(trainingFilesItem);
    }

    return this;
  }
  /**
   * The Unix timestamp (in seconds) for when the fine-tuning job was last updated.
   **/
  public FineTune updatedAt(Integer updatedAt) {
    this.updatedAt = updatedAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the fine-tuning job was last updated.")
  @JsonProperty("updated_at")
  @NotNull
  public Integer getUpdatedAt() {
    return updatedAt;
  }

  @JsonProperty("updated_at")
  public void setUpdatedAt(Integer updatedAt) {
    this.updatedAt = updatedAt;
  }

  /**
   * The list of files used for validation.
   **/
  public FineTune validationFiles(List<@Valid OpenAIFile> validationFiles) {
    this.validationFiles = validationFiles;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The list of files used for validation.")
  @JsonProperty("validation_files")
  @NotNull
  public List<OpenAIFile> getValidationFiles() {
    return validationFiles;
  }

  @JsonProperty("validation_files")
  public void setValidationFiles(List<@Valid OpenAIFile> validationFiles) {
    this.validationFiles = validationFiles;
  }

  public FineTune addValidationFilesItem(OpenAIFile validationFilesItem) {
    if (this.validationFiles == null) {
      this.validationFiles = new ArrayList<>();
    }

    this.validationFiles.add(validationFilesItem);
    return this;
  }

  public FineTune removeValidationFilesItem(OpenAIFile validationFilesItem) {
    if (validationFilesItem != null && this.validationFiles != null) {
      this.validationFiles.remove(validationFilesItem);
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
    FineTune fineTune = (FineTune) o;
    return Objects.equals(this.id, fineTune.id) &&
        Objects.equals(this.createdAt, fineTune.createdAt) &&
        Objects.equals(this.events, fineTune.events) &&
        Objects.equals(this.fineTunedModel, fineTune.fineTunedModel) &&
        Objects.equals(this.hyperparams, fineTune.hyperparams) &&
        Objects.equals(this.model, fineTune.model) &&
        Objects.equals(this._object, fineTune._object) &&
        Objects.equals(this.organizationId, fineTune.organizationId) &&
        Objects.equals(this.resultFiles, fineTune.resultFiles) &&
        Objects.equals(this.status, fineTune.status) &&
        Objects.equals(this.trainingFiles, fineTune.trainingFiles) &&
        Objects.equals(this.updatedAt, fineTune.updatedAt) &&
        Objects.equals(this.validationFiles, fineTune.validationFiles);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, createdAt, events, fineTunedModel, hyperparams, model, _object, organizationId, resultFiles, status, trainingFiles, updatedAt, validationFiles);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class FineTune {\n");
    
    sb.append("    id: ").append(toIndentedString(id)).append("\n");
    sb.append("    createdAt: ").append(toIndentedString(createdAt)).append("\n");
    sb.append("    events: ").append(toIndentedString(events)).append("\n");
    sb.append("    fineTunedModel: ").append(toIndentedString(fineTunedModel)).append("\n");
    sb.append("    hyperparams: ").append(toIndentedString(hyperparams)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    _object: ").append(toIndentedString(_object)).append("\n");
    sb.append("    organizationId: ").append(toIndentedString(organizationId)).append("\n");
    sb.append("    resultFiles: ").append(toIndentedString(resultFiles)).append("\n");
    sb.append("    status: ").append(toIndentedString(status)).append("\n");
    sb.append("    trainingFiles: ").append(toIndentedString(trainingFiles)).append("\n");
    sb.append("    updatedAt: ").append(toIndentedString(updatedAt)).append("\n");
    sb.append("    validationFiles: ").append(toIndentedString(validationFiles)).append("\n");
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

