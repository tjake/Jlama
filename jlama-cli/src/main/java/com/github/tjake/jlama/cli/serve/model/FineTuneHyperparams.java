package com.github.tjake.jlama.cli.serve.model;

import com.fasterxml.jackson.annotation.JsonTypeName;
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

/**
 * The hyperparameters used for the fine-tuning job. See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/hyperparameters) for more details.
 **/
@ApiModel(description = "The hyperparameters used for the fine-tuning job. See the [fine-tuning guide](/docs/guides/legacy-fine-tuning/hyperparameters) for more details.")
@JsonTypeName("FineTune_hyperparams")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class FineTuneHyperparams   {
  private @Valid Integer batchSize;
  private @Valid Integer classificationNClasses;
  private @Valid String classificationPositiveClass;
  private @Valid Boolean computeClassificationMetrics;
  private @Valid BigDecimal learningRateMultiplier;
  private @Valid Integer nEpochs;
  private @Valid BigDecimal promptLossWeight;

  /**
   * The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass. 
   **/
  public FineTuneHyperparams batchSize(Integer batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass. ")
  @JsonProperty("batch_size")
  @NotNull
  public Integer getBatchSize() {
    return batchSize;
  }

  @JsonProperty("batch_size")
  public void setBatchSize(Integer batchSize) {
    this.batchSize = batchSize;
  }

  /**
   * The number of classes to use for computing classification metrics. 
   **/
  public FineTuneHyperparams classificationNClasses(Integer classificationNClasses) {
    this.classificationNClasses = classificationNClasses;
    return this;
  }

  
  @ApiModelProperty(value = "The number of classes to use for computing classification metrics. ")
  @JsonProperty("classification_n_classes")
  public Integer getClassificationNClasses() {
    return classificationNClasses;
  }

  @JsonProperty("classification_n_classes")
  public void setClassificationNClasses(Integer classificationNClasses) {
    this.classificationNClasses = classificationNClasses;
  }

  /**
   * The positive class to use for computing classification metrics. 
   **/
  public FineTuneHyperparams classificationPositiveClass(String classificationPositiveClass) {
    this.classificationPositiveClass = classificationPositiveClass;
    return this;
  }

  
  @ApiModelProperty(value = "The positive class to use for computing classification metrics. ")
  @JsonProperty("classification_positive_class")
  public String getClassificationPositiveClass() {
    return classificationPositiveClass;
  }

  @JsonProperty("classification_positive_class")
  public void setClassificationPositiveClass(String classificationPositiveClass) {
    this.classificationPositiveClass = classificationPositiveClass;
  }

  /**
   * The classification metrics to compute using the validation dataset at the end of every epoch. 
   **/
  public FineTuneHyperparams computeClassificationMetrics(Boolean computeClassificationMetrics) {
    this.computeClassificationMetrics = computeClassificationMetrics;
    return this;
  }

  
  @ApiModelProperty(value = "The classification metrics to compute using the validation dataset at the end of every epoch. ")
  @JsonProperty("compute_classification_metrics")
  public Boolean getComputeClassificationMetrics() {
    return computeClassificationMetrics;
  }

  @JsonProperty("compute_classification_metrics")
  public void setComputeClassificationMetrics(Boolean computeClassificationMetrics) {
    this.computeClassificationMetrics = computeClassificationMetrics;
  }

  /**
   * The learning rate multiplier to use for training. 
   **/
  public FineTuneHyperparams learningRateMultiplier(BigDecimal learningRateMultiplier) {
    this.learningRateMultiplier = learningRateMultiplier;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The learning rate multiplier to use for training. ")
  @JsonProperty("learning_rate_multiplier")
  @NotNull
  public BigDecimal getLearningRateMultiplier() {
    return learningRateMultiplier;
  }

  @JsonProperty("learning_rate_multiplier")
  public void setLearningRateMultiplier(BigDecimal learningRateMultiplier) {
    this.learningRateMultiplier = learningRateMultiplier;
  }

  /**
   * The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset. 
   **/
  public FineTuneHyperparams nEpochs(Integer nEpochs) {
    this.nEpochs = nEpochs;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset. ")
  @JsonProperty("n_epochs")
  @NotNull
  public Integer getnEpochs() {
    return nEpochs;
  }

  @JsonProperty("n_epochs")
  public void setnEpochs(Integer nEpochs) {
    this.nEpochs = nEpochs;
  }

  /**
   * The weight to use for loss on the prompt tokens. 
   **/
  public FineTuneHyperparams promptLossWeight(BigDecimal promptLossWeight) {
    this.promptLossWeight = promptLossWeight;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The weight to use for loss on the prompt tokens. ")
  @JsonProperty("prompt_loss_weight")
  @NotNull
  public BigDecimal getPromptLossWeight() {
    return promptLossWeight;
  }

  @JsonProperty("prompt_loss_weight")
  public void setPromptLossWeight(BigDecimal promptLossWeight) {
    this.promptLossWeight = promptLossWeight;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    FineTuneHyperparams fineTuneHyperparams = (FineTuneHyperparams) o;
    return Objects.equals(this.batchSize, fineTuneHyperparams.batchSize) &&
        Objects.equals(this.classificationNClasses, fineTuneHyperparams.classificationNClasses) &&
        Objects.equals(this.classificationPositiveClass, fineTuneHyperparams.classificationPositiveClass) &&
        Objects.equals(this.computeClassificationMetrics, fineTuneHyperparams.computeClassificationMetrics) &&
        Objects.equals(this.learningRateMultiplier, fineTuneHyperparams.learningRateMultiplier) &&
        Objects.equals(this.nEpochs, fineTuneHyperparams.nEpochs) &&
        Objects.equals(this.promptLossWeight, fineTuneHyperparams.promptLossWeight);
  }

  @Override
  public int hashCode() {
    return Objects.hash(batchSize, classificationNClasses, classificationPositiveClass, computeClassificationMetrics, learningRateMultiplier, nEpochs, promptLossWeight);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class FineTuneHyperparams {\n");
    
    sb.append("    batchSize: ").append(toIndentedString(batchSize)).append("\n");
    sb.append("    classificationNClasses: ").append(toIndentedString(classificationNClasses)).append("\n");
    sb.append("    classificationPositiveClass: ").append(toIndentedString(classificationPositiveClass)).append("\n");
    sb.append("    computeClassificationMetrics: ").append(toIndentedString(computeClassificationMetrics)).append("\n");
    sb.append("    learningRateMultiplier: ").append(toIndentedString(learningRateMultiplier)).append("\n");
    sb.append("    nEpochs: ").append(toIndentedString(nEpochs)).append("\n");
    sb.append("    promptLossWeight: ").append(toIndentedString(promptLossWeight)).append("\n");
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

