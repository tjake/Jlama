package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CreateFineTuneRequestHyperparameters;
import com.github.tjake.jlama.cli.serve.model.CreateFineTuneRequestModel;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import java.math.BigDecimal;
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



@JsonTypeName("CreateFineTuneRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateFineTuneRequest   {
  private @Valid String trainingFile;
  private @Valid Integer batchSize;
  private @Valid List<BigDecimal> classificationBetas;
  private @Valid Integer classificationNClasses;
  private @Valid String classificationPositiveClass;
  private @Valid Boolean computeClassificationMetrics = false;
  private @Valid CreateFineTuneRequestHyperparameters hyperparameters;
  private @Valid BigDecimal learningRateMultiplier;
  private @Valid CreateFineTuneRequestModel model;
  private @Valid BigDecimal promptLossWeight = new BigDecimal("0.01");
  private @Valid String suffix;
  private @Valid String validationFile;

  /**
   * The ID of an uploaded file that contains training data.  See [upload file](/docs/api-reference/files/upload) for how to upload a file.  Your dataset must be formatted as a JSONL file, where each training example is a JSON object with the keys \&quot;prompt\&quot; and \&quot;completion\&quot;. Additionally, you must upload your file with the purpose &#x60;fine-tune&#x60;.  See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/creating-training-data) for more details. 
   **/
  public CreateFineTuneRequest trainingFile(String trainingFile) {
    this.trainingFile = trainingFile;
    return this;
  }

  
  @ApiModelProperty(example = "file-abc123", required = true, value = "The ID of an uploaded file that contains training data.  See [upload file](/docs/api-reference/files/upload) for how to upload a file.  Your dataset must be formatted as a JSONL file, where each training example is a JSON object with the keys \"prompt\" and \"completion\". Additionally, you must upload your file with the purpose `fine-tune`.  See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/creating-training-data) for more details. ")
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
   * The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass.  By default, the batch size will be dynamically configured to be ~0.2% of the number of examples in the training set, capped at 256 - in general, we&#39;ve found that larger batch sizes tend to work better for larger datasets. 
   **/
  public CreateFineTuneRequest batchSize(Integer batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  
  @ApiModelProperty(value = "The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass.  By default, the batch size will be dynamically configured to be ~0.2% of the number of examples in the training set, capped at 256 - in general, we've found that larger batch sizes tend to work better for larger datasets. ")
  @JsonProperty("batch_size")
  public Integer getBatchSize() {
    return batchSize;
  }

  @JsonProperty("batch_size")
  public void setBatchSize(Integer batchSize) {
    this.batchSize = batchSize;
  }

  /**
   * If this is provided, we calculate F-beta scores at the specified beta values. The F-beta score is a generalization of F-1 score. This is only used for binary classification.  With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight. A larger beta score puts more weight on recall and less on precision. A smaller beta score puts more weight on precision and less on recall. 
   **/
  public CreateFineTuneRequest classificationBetas(List<BigDecimal> classificationBetas) {
    this.classificationBetas = classificationBetas;
    return this;
  }

  
  @ApiModelProperty(example = "[0.6,1,1.5,2]", value = "If this is provided, we calculate F-beta scores at the specified beta values. The F-beta score is a generalization of F-1 score. This is only used for binary classification.  With a beta of 1 (i.e. the F-1 score), precision and recall are given the same weight. A larger beta score puts more weight on recall and less on precision. A smaller beta score puts more weight on precision and less on recall. ")
  @JsonProperty("classification_betas")
  public List<BigDecimal> getClassificationBetas() {
    return classificationBetas;
  }

  @JsonProperty("classification_betas")
  public void setClassificationBetas(List<BigDecimal> classificationBetas) {
    this.classificationBetas = classificationBetas;
  }

  public CreateFineTuneRequest addClassificationBetasItem(BigDecimal classificationBetasItem) {
    if (this.classificationBetas == null) {
      this.classificationBetas = new ArrayList<>();
    }

    this.classificationBetas.add(classificationBetasItem);
    return this;
  }

  public CreateFineTuneRequest removeClassificationBetasItem(BigDecimal classificationBetasItem) {
    if (classificationBetasItem != null && this.classificationBetas != null) {
      this.classificationBetas.remove(classificationBetasItem);
    }

    return this;
  }
  /**
   * The number of classes in a classification task.  This parameter is required for multiclass classification. 
   **/
  public CreateFineTuneRequest classificationNClasses(Integer classificationNClasses) {
    this.classificationNClasses = classificationNClasses;
    return this;
  }

  
  @ApiModelProperty(value = "The number of classes in a classification task.  This parameter is required for multiclass classification. ")
  @JsonProperty("classification_n_classes")
  public Integer getClassificationNClasses() {
    return classificationNClasses;
  }

  @JsonProperty("classification_n_classes")
  public void setClassificationNClasses(Integer classificationNClasses) {
    this.classificationNClasses = classificationNClasses;
  }

  /**
   * The positive class in binary classification.  This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification. 
   **/
  public CreateFineTuneRequest classificationPositiveClass(String classificationPositiveClass) {
    this.classificationPositiveClass = classificationPositiveClass;
    return this;
  }

  
  @ApiModelProperty(value = "The positive class in binary classification.  This parameter is needed to generate precision, recall, and F1 metrics when doing binary classification. ")
  @JsonProperty("classification_positive_class")
  public String getClassificationPositiveClass() {
    return classificationPositiveClass;
  }

  @JsonProperty("classification_positive_class")
  public void setClassificationPositiveClass(String classificationPositiveClass) {
    this.classificationPositiveClass = classificationPositiveClass;
  }

  /**
   * If set, we calculate classification-specific metrics such as accuracy and F-1 score using the validation set at the end of every epoch. These metrics can be viewed in the [results file](/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model).  In order to compute classification metrics, you must provide a &#x60;validation_file&#x60;. Additionally, you must specify &#x60;classification_n_classes&#x60; for multiclass classification or &#x60;classification_positive_class&#x60; for binary classification. 
   **/
  public CreateFineTuneRequest computeClassificationMetrics(Boolean computeClassificationMetrics) {
    this.computeClassificationMetrics = computeClassificationMetrics;
    return this;
  }

  
  @ApiModelProperty(value = "If set, we calculate classification-specific metrics such as accuracy and F-1 score using the validation set at the end of every epoch. These metrics can be viewed in the [results file](/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model).  In order to compute classification metrics, you must provide a `validation_file`. Additionally, you must specify `classification_n_classes` for multiclass classification or `classification_positive_class` for binary classification. ")
  @JsonProperty("compute_classification_metrics")
  public Boolean getComputeClassificationMetrics() {
    return computeClassificationMetrics;
  }

  @JsonProperty("compute_classification_metrics")
  public void setComputeClassificationMetrics(Boolean computeClassificationMetrics) {
    this.computeClassificationMetrics = computeClassificationMetrics;
  }

  /**
   **/
  public CreateFineTuneRequest hyperparameters(CreateFineTuneRequestHyperparameters hyperparameters) {
    this.hyperparameters = hyperparameters;
    return this;
  }

  
  @ApiModelProperty(value = "")
  @JsonProperty("hyperparameters")
  public CreateFineTuneRequestHyperparameters getHyperparameters() {
    return hyperparameters;
  }

  @JsonProperty("hyperparameters")
  public void setHyperparameters(CreateFineTuneRequestHyperparameters hyperparameters) {
    this.hyperparameters = hyperparameters;
  }

  /**
   * The learning rate multiplier to use for training. The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this value.  By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on final &#x60;batch_size&#x60; (larger learning rates tend to perform better with larger batch sizes). We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results. 
   **/
  public CreateFineTuneRequest learningRateMultiplier(BigDecimal learningRateMultiplier) {
    this.learningRateMultiplier = learningRateMultiplier;
    return this;
  }

  
  @ApiModelProperty(value = "The learning rate multiplier to use for training. The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this value.  By default, the learning rate multiplier is the 0.05, 0.1, or 0.2 depending on final `batch_size` (larger learning rates tend to perform better with larger batch sizes). We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results. ")
  @JsonProperty("learning_rate_multiplier")
  public BigDecimal getLearningRateMultiplier() {
    return learningRateMultiplier;
  }

  @JsonProperty("learning_rate_multiplier")
  public void setLearningRateMultiplier(BigDecimal learningRateMultiplier) {
    this.learningRateMultiplier = learningRateMultiplier;
  }

  /**
   **/
  public CreateFineTuneRequest model(CreateFineTuneRequestModel model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(value = "")
  @JsonProperty("model")
  public CreateFineTuneRequestModel getModel() {
    return model;
  }

  @JsonProperty("model")
  public void setModel(CreateFineTuneRequestModel model) {
    this.model = model;
  }

  /**
   * The weight to use for loss on the prompt tokens. This controls how much the model tries to learn to generate the prompt (as compared to the completion which always has a weight of 1.0), and can add a stabilizing effect to training when completions are short.  If prompts are extremely long (relative to completions), it may make sense to reduce this weight so as to avoid over-prioritizing learning the prompt. 
   **/
  public CreateFineTuneRequest promptLossWeight(BigDecimal promptLossWeight) {
    this.promptLossWeight = promptLossWeight;
    return this;
  }

  
  @ApiModelProperty(value = "The weight to use for loss on the prompt tokens. This controls how much the model tries to learn to generate the prompt (as compared to the completion which always has a weight of 1.0), and can add a stabilizing effect to training when completions are short.  If prompts are extremely long (relative to completions), it may make sense to reduce this weight so as to avoid over-prioritizing learning the prompt. ")
  @JsonProperty("prompt_loss_weight")
  public BigDecimal getPromptLossWeight() {
    return promptLossWeight;
  }

  @JsonProperty("prompt_loss_weight")
  public void setPromptLossWeight(BigDecimal promptLossWeight) {
    this.promptLossWeight = promptLossWeight;
  }

  /**
   * A string of up to 40 characters that will be added to your fine-tuned model name.  For example, a &#x60;suffix&#x60; of \&quot;custom-model-name\&quot; would produce a model name like &#x60;ada:ft-your-org:custom-model-name-2022-02-15-04-21-04&#x60;. 
   **/
  public CreateFineTuneRequest suffix(String suffix) {
    this.suffix = suffix;
    return this;
  }

  
  @ApiModelProperty(value = "A string of up to 40 characters that will be added to your fine-tuned model name.  For example, a `suffix` of \"custom-model-name\" would produce a model name like `ada:ft-your-org:custom-model-name-2022-02-15-04-21-04`. ")
  @JsonProperty("suffix")
 @Size(min=1,max=40)  public String getSuffix() {
    return suffix;
  }

  @JsonProperty("suffix")
  public void setSuffix(String suffix) {
    this.suffix = suffix;
  }

  /**
   * The ID of an uploaded file that contains validation data.  If you provide this file, the data is used to generate validation metrics periodically during fine-tuning. These metrics can be viewed in the [fine-tuning results file](/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model). Your train and validation data should be mutually exclusive.  Your dataset must be formatted as a JSONL file, where each validation example is a JSON object with the keys \&quot;prompt\&quot; and \&quot;completion\&quot;. Additionally, you must upload your file with the purpose &#x60;fine-tune&#x60;.  See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/creating-training-data) for more details. 
   **/
  public CreateFineTuneRequest validationFile(String validationFile) {
    this.validationFile = validationFile;
    return this;
  }

  
  @ApiModelProperty(example = "file-abc123", value = "The ID of an uploaded file that contains validation data.  If you provide this file, the data is used to generate validation metrics periodically during fine-tuning. These metrics can be viewed in the [fine-tuning results file](/docs/guides/legacy-fine-tuning/analyzing-your-fine-tuned-model). Your train and validation data should be mutually exclusive.  Your dataset must be formatted as a JSONL file, where each validation example is a JSON object with the keys \"prompt\" and \"completion\". Additionally, you must upload your file with the purpose `fine-tune`.  See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/creating-training-data) for more details. ")
  @JsonProperty("validation_file")
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
    CreateFineTuneRequest createFineTuneRequest = (CreateFineTuneRequest) o;
    return Objects.equals(this.trainingFile, createFineTuneRequest.trainingFile) &&
        Objects.equals(this.batchSize, createFineTuneRequest.batchSize) &&
        Objects.equals(this.classificationBetas, createFineTuneRequest.classificationBetas) &&
        Objects.equals(this.classificationNClasses, createFineTuneRequest.classificationNClasses) &&
        Objects.equals(this.classificationPositiveClass, createFineTuneRequest.classificationPositiveClass) &&
        Objects.equals(this.computeClassificationMetrics, createFineTuneRequest.computeClassificationMetrics) &&
        Objects.equals(this.hyperparameters, createFineTuneRequest.hyperparameters) &&
        Objects.equals(this.learningRateMultiplier, createFineTuneRequest.learningRateMultiplier) &&
        Objects.equals(this.model, createFineTuneRequest.model) &&
        Objects.equals(this.promptLossWeight, createFineTuneRequest.promptLossWeight) &&
        Objects.equals(this.suffix, createFineTuneRequest.suffix) &&
        Objects.equals(this.validationFile, createFineTuneRequest.validationFile);
  }

  @Override
  public int hashCode() {
    return Objects.hash(trainingFile, batchSize, classificationBetas, classificationNClasses, classificationPositiveClass, computeClassificationMetrics, hyperparameters, learningRateMultiplier, model, promptLossWeight, suffix, validationFile);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateFineTuneRequest {\n");
    
    sb.append("    trainingFile: ").append(toIndentedString(trainingFile)).append("\n");
    sb.append("    batchSize: ").append(toIndentedString(batchSize)).append("\n");
    sb.append("    classificationBetas: ").append(toIndentedString(classificationBetas)).append("\n");
    sb.append("    classificationNClasses: ").append(toIndentedString(classificationNClasses)).append("\n");
    sb.append("    classificationPositiveClass: ").append(toIndentedString(classificationPositiveClass)).append("\n");
    sb.append("    computeClassificationMetrics: ").append(toIndentedString(computeClassificationMetrics)).append("\n");
    sb.append("    hyperparameters: ").append(toIndentedString(hyperparameters)).append("\n");
    sb.append("    learningRateMultiplier: ").append(toIndentedString(learningRateMultiplier)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    promptLossWeight: ").append(toIndentedString(promptLossWeight)).append("\n");
    sb.append("    suffix: ").append(toIndentedString(suffix)).append("\n");
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

