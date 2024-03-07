package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.FineTuningJobError;
import com.github.tjake.jlama.cli.serve.model.FineTuningJobHyperparameters;
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
 * The &#x60;fine_tuning.job&#x60; object represents a fine-tuning job that has been created through the API. 
 **/
@ApiModel(description = "The `fine_tuning.job` object represents a fine-tuning job that has been created through the API. ")
@JsonTypeName("FineTuningJob")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class FineTuningJob   {
  private @Valid String id;
  private @Valid Integer createdAt;
  private @Valid FineTuningJobError error;
  private @Valid String fineTunedModel;
  private @Valid Integer finishedAt;
  private @Valid FineTuningJobHyperparameters hyperparameters;
  private @Valid String model;
  public enum ObjectEnum {

    FINE_TUNING_JOB(String.valueOf("fine_tuning.job"));


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
  private @Valid List<String> resultFiles = new ArrayList<>();
  public enum StatusEnum {

    VALIDATING_FILES(String.valueOf("validating_files")), QUEUED(String.valueOf("queued")), RUNNING(String.valueOf("running")), SUCCEEDED(String.valueOf("succeeded")), FAILED(String.valueOf("failed")), CANCELLED(String.valueOf("cancelled"));


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
  private @Valid Integer trainedTokens;
  private @Valid String trainingFile;
  private @Valid String validationFile;

  /**
   * The object identifier, which can be referenced in the API endpoints.
   **/
  public FineTuningJob id(String id) {
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
  public FineTuningJob createdAt(Integer createdAt) {
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
   **/
  public FineTuningJob error(FineTuningJobError error) {
    this.error = error;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("error")
  @NotNull
  public FineTuningJobError getError() {
    return error;
  }

  @JsonProperty("error")
  public void setError(FineTuningJobError error) {
    this.error = error;
  }

  /**
   * The name of the fine-tuned model that is being created. The value will be null if the fine-tuning job is still running.
   **/
  public FineTuningJob fineTunedModel(String fineTunedModel) {
    this.fineTunedModel = fineTunedModel;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The name of the fine-tuned model that is being created. The value will be null if the fine-tuning job is still running.")
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
   * The Unix timestamp (in seconds) for when the fine-tuning job was finished. The value will be null if the fine-tuning job is still running.
   **/
  public FineTuningJob finishedAt(Integer finishedAt) {
    this.finishedAt = finishedAt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) for when the fine-tuning job was finished. The value will be null if the fine-tuning job is still running.")
  @JsonProperty("finished_at")
  @NotNull
  public Integer getFinishedAt() {
    return finishedAt;
  }

  @JsonProperty("finished_at")
  public void setFinishedAt(Integer finishedAt) {
    this.finishedAt = finishedAt;
  }

  /**
   **/
  public FineTuningJob hyperparameters(FineTuningJobHyperparameters hyperparameters) {
    this.hyperparameters = hyperparameters;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("hyperparameters")
  @NotNull
  public FineTuningJobHyperparameters getHyperparameters() {
    return hyperparameters;
  }

  @JsonProperty("hyperparameters")
  public void setHyperparameters(FineTuningJobHyperparameters hyperparameters) {
    this.hyperparameters = hyperparameters;
  }

  /**
   * The base model that is being fine-tuned.
   **/
  public FineTuningJob model(String model) {
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
   * The object type, which is always \&quot;fine_tuning.job\&quot;.
   **/
  public FineTuningJob _object(ObjectEnum _object) {
    this._object = _object;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The object type, which is always \"fine_tuning.job\".")
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
  public FineTuningJob organizationId(String organizationId) {
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
   * The compiled results file ID(s) for the fine-tuning job. You can retrieve the results with the [Files API](/docs/api-reference/files/retrieve-contents).
   **/
  public FineTuningJob resultFiles(List<String> resultFiles) {
    this.resultFiles = resultFiles;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The compiled results file ID(s) for the fine-tuning job. You can retrieve the results with the [Files API](/docs/api-reference/files/retrieve-contents).")
  @JsonProperty("result_files")
  @NotNull
  public List<String> getResultFiles() {
    return resultFiles;
  }

  @JsonProperty("result_files")
  public void setResultFiles(List<String> resultFiles) {
    this.resultFiles = resultFiles;
  }

  public FineTuningJob addResultFilesItem(String resultFilesItem) {
    if (this.resultFiles == null) {
      this.resultFiles = new ArrayList<>();
    }

    this.resultFiles.add(resultFilesItem);
    return this;
  }

  public FineTuningJob removeResultFilesItem(String resultFilesItem) {
    if (resultFilesItem != null && this.resultFiles != null) {
      this.resultFiles.remove(resultFilesItem);
    }

    return this;
  }
  /**
   * The current status of the fine-tuning job, which can be either &#x60;validating_files&#x60;, &#x60;queued&#x60;, &#x60;running&#x60;, &#x60;succeeded&#x60;, &#x60;failed&#x60;, or &#x60;cancelled&#x60;.
   **/
  public FineTuningJob status(StatusEnum status) {
    this.status = status;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The current status of the fine-tuning job, which can be either `validating_files`, `queued`, `running`, `succeeded`, `failed`, or `cancelled`.")
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
   * The total number of billable tokens processed by this fine-tuning job. The value will be null if the fine-tuning job is still running.
   **/
  public FineTuningJob trainedTokens(Integer trainedTokens) {
    this.trainedTokens = trainedTokens;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The total number of billable tokens processed by this fine-tuning job. The value will be null if the fine-tuning job is still running.")
  @JsonProperty("trained_tokens")
  @NotNull
  public Integer getTrainedTokens() {
    return trainedTokens;
  }

  @JsonProperty("trained_tokens")
  public void setTrainedTokens(Integer trainedTokens) {
    this.trainedTokens = trainedTokens;
  }

  /**
   * The file ID used for training. You can retrieve the training data with the [Files API](/docs/api-reference/files/retrieve-contents).
   **/
  public FineTuningJob trainingFile(String trainingFile) {
    this.trainingFile = trainingFile;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The file ID used for training. You can retrieve the training data with the [Files API](/docs/api-reference/files/retrieve-contents).")
  @JsonProperty("training_file")
  @NotNull
  public String getTrainingFile() {
    return trainingFile;
  }

  @JsonProperty("training_file")
  public void setTrainingFile(String trainingFile) {
    this.trainingFile = trainingFile;
  }

  /**
   * The file ID used for validation. You can retrieve the validation results with the [Files API](/docs/api-reference/files/retrieve-contents).
   **/
  public FineTuningJob validationFile(String validationFile) {
    this.validationFile = validationFile;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The file ID used for validation. You can retrieve the validation results with the [Files API](/docs/api-reference/files/retrieve-contents).")
  @JsonProperty("validation_file")
  @NotNull
  public String getValidationFile() {
    return validationFile;
  }

  @JsonProperty("validation_file")
  public void setValidationFile(String validationFile) {
    this.validationFile = validationFile;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    FineTuningJob fineTuningJob = (FineTuningJob) o;
    return Objects.equals(this.id, fineTuningJob.id) &&
        Objects.equals(this.createdAt, fineTuningJob.createdAt) &&
        Objects.equals(this.error, fineTuningJob.error) &&
        Objects.equals(this.fineTunedModel, fineTuningJob.fineTunedModel) &&
        Objects.equals(this.finishedAt, fineTuningJob.finishedAt) &&
        Objects.equals(this.hyperparameters, fineTuningJob.hyperparameters) &&
        Objects.equals(this.model, fineTuningJob.model) &&
        Objects.equals(this._object, fineTuningJob._object) &&
        Objects.equals(this.organizationId, fineTuningJob.organizationId) &&
        Objects.equals(this.resultFiles, fineTuningJob.resultFiles) &&
        Objects.equals(this.status, fineTuningJob.status) &&
        Objects.equals(this.trainedTokens, fineTuningJob.trainedTokens) &&
        Objects.equals(this.trainingFile, fineTuningJob.trainingFile) &&
        Objects.equals(this.validationFile, fineTuningJob.validationFile);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, createdAt, error, fineTunedModel, finishedAt, hyperparameters, model, _object, organizationId, resultFiles, status, trainedTokens, trainingFile, validationFile);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class FineTuningJob {\n");
    
    sb.append("    id: ").append(toIndentedString(id)).append("\n");
    sb.append("    createdAt: ").append(toIndentedString(createdAt)).append("\n");
    sb.append("    error: ").append(toIndentedString(error)).append("\n");
    sb.append("    fineTunedModel: ").append(toIndentedString(fineTunedModel)).append("\n");
    sb.append("    finishedAt: ").append(toIndentedString(finishedAt)).append("\n");
    sb.append("    hyperparameters: ").append(toIndentedString(hyperparameters)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    _object: ").append(toIndentedString(_object)).append("\n");
    sb.append("    organizationId: ").append(toIndentedString(organizationId)).append("\n");
    sb.append("    resultFiles: ").append(toIndentedString(resultFiles)).append("\n");
    sb.append("    status: ").append(toIndentedString(status)).append("\n");
    sb.append("    trainedTokens: ").append(toIndentedString(trainedTokens)).append("\n");
    sb.append("    trainingFile: ").append(toIndentedString(trainingFile)).append("\n");
    sb.append("    validationFile: ").append(toIndentedString(validationFile)).append("\n");
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

