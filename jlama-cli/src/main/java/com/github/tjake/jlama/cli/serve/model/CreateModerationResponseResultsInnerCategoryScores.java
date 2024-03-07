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
 * A list of the categories along with their scores as predicted by model.
 **/
@ApiModel(description = "A list of the categories along with their scores as predicted by model.")
@JsonTypeName("CreateModerationResponse_results_inner_category_scores")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateModerationResponseResultsInnerCategoryScores   {
  private @Valid BigDecimal hate;
  private @Valid BigDecimal hateThreatening;
  private @Valid BigDecimal harassment;
  private @Valid BigDecimal harassmentThreatening;
  private @Valid BigDecimal selfHarm;
  private @Valid BigDecimal selfHarmIntent;
  private @Valid BigDecimal selfHarmInstructions;
  private @Valid BigDecimal sexual;
  private @Valid BigDecimal sexualMinors;
  private @Valid BigDecimal violence;
  private @Valid BigDecimal violenceGraphic;

  /**
   * The score for the category &#39;hate&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores hate(BigDecimal hate) {
    this.hate = hate;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'hate'.")
  @JsonProperty("hate")
  @NotNull
  public BigDecimal getHate() {
    return hate;
  }

  @JsonProperty("hate")
  public void setHate(BigDecimal hate) {
    this.hate = hate;
  }

  /**
   * The score for the category &#39;hate/threatening&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores hateThreatening(BigDecimal hateThreatening) {
    this.hateThreatening = hateThreatening;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'hate/threatening'.")
  @JsonProperty("hate/threatening")
  @NotNull
  public BigDecimal getHateThreatening() {
    return hateThreatening;
  }

  @JsonProperty("hate/threatening")
  public void setHateThreatening(BigDecimal hateThreatening) {
    this.hateThreatening = hateThreatening;
  }

  /**
   * The score for the category &#39;harassment&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores harassment(BigDecimal harassment) {
    this.harassment = harassment;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'harassment'.")
  @JsonProperty("harassment")
  @NotNull
  public BigDecimal getHarassment() {
    return harassment;
  }

  @JsonProperty("harassment")
  public void setHarassment(BigDecimal harassment) {
    this.harassment = harassment;
  }

  /**
   * The score for the category &#39;harassment/threatening&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores harassmentThreatening(BigDecimal harassmentThreatening) {
    this.harassmentThreatening = harassmentThreatening;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'harassment/threatening'.")
  @JsonProperty("harassment/threatening")
  @NotNull
  public BigDecimal getHarassmentThreatening() {
    return harassmentThreatening;
  }

  @JsonProperty("harassment/threatening")
  public void setHarassmentThreatening(BigDecimal harassmentThreatening) {
    this.harassmentThreatening = harassmentThreatening;
  }

  /**
   * The score for the category &#39;self-harm&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores selfHarm(BigDecimal selfHarm) {
    this.selfHarm = selfHarm;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'self-harm'.")
  @JsonProperty("self-harm")
  @NotNull
  public BigDecimal getSelfHarm() {
    return selfHarm;
  }

  @JsonProperty("self-harm")
  public void setSelfHarm(BigDecimal selfHarm) {
    this.selfHarm = selfHarm;
  }

  /**
   * The score for the category &#39;self-harm/intent&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores selfHarmIntent(BigDecimal selfHarmIntent) {
    this.selfHarmIntent = selfHarmIntent;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'self-harm/intent'.")
  @JsonProperty("self-harm/intent")
  @NotNull
  public BigDecimal getSelfHarmIntent() {
    return selfHarmIntent;
  }

  @JsonProperty("self-harm/intent")
  public void setSelfHarmIntent(BigDecimal selfHarmIntent) {
    this.selfHarmIntent = selfHarmIntent;
  }

  /**
   * The score for the category &#39;self-harm/instructions&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores selfHarmInstructions(BigDecimal selfHarmInstructions) {
    this.selfHarmInstructions = selfHarmInstructions;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'self-harm/instructions'.")
  @JsonProperty("self-harm/instructions")
  @NotNull
  public BigDecimal getSelfHarmInstructions() {
    return selfHarmInstructions;
  }

  @JsonProperty("self-harm/instructions")
  public void setSelfHarmInstructions(BigDecimal selfHarmInstructions) {
    this.selfHarmInstructions = selfHarmInstructions;
  }

  /**
   * The score for the category &#39;sexual&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores sexual(BigDecimal sexual) {
    this.sexual = sexual;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'sexual'.")
  @JsonProperty("sexual")
  @NotNull
  public BigDecimal getSexual() {
    return sexual;
  }

  @JsonProperty("sexual")
  public void setSexual(BigDecimal sexual) {
    this.sexual = sexual;
  }

  /**
   * The score for the category &#39;sexual/minors&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores sexualMinors(BigDecimal sexualMinors) {
    this.sexualMinors = sexualMinors;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'sexual/minors'.")
  @JsonProperty("sexual/minors")
  @NotNull
  public BigDecimal getSexualMinors() {
    return sexualMinors;
  }

  @JsonProperty("sexual/minors")
  public void setSexualMinors(BigDecimal sexualMinors) {
    this.sexualMinors = sexualMinors;
  }

  /**
   * The score for the category &#39;violence&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores violence(BigDecimal violence) {
    this.violence = violence;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'violence'.")
  @JsonProperty("violence")
  @NotNull
  public BigDecimal getViolence() {
    return violence;
  }

  @JsonProperty("violence")
  public void setViolence(BigDecimal violence) {
    this.violence = violence;
  }

  /**
   * The score for the category &#39;violence/graphic&#39;.
   **/
  public CreateModerationResponseResultsInnerCategoryScores violenceGraphic(BigDecimal violenceGraphic) {
    this.violenceGraphic = violenceGraphic;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The score for the category 'violence/graphic'.")
  @JsonProperty("violence/graphic")
  @NotNull
  public BigDecimal getViolenceGraphic() {
    return violenceGraphic;
  }

  @JsonProperty("violence/graphic")
  public void setViolenceGraphic(BigDecimal violenceGraphic) {
    this.violenceGraphic = violenceGraphic;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateModerationResponseResultsInnerCategoryScores createModerationResponseResultsInnerCategoryScores = (CreateModerationResponseResultsInnerCategoryScores) o;
    return Objects.equals(this.hate, createModerationResponseResultsInnerCategoryScores.hate) &&
        Objects.equals(this.hateThreatening, createModerationResponseResultsInnerCategoryScores.hateThreatening) &&
        Objects.equals(this.harassment, createModerationResponseResultsInnerCategoryScores.harassment) &&
        Objects.equals(this.harassmentThreatening, createModerationResponseResultsInnerCategoryScores.harassmentThreatening) &&
        Objects.equals(this.selfHarm, createModerationResponseResultsInnerCategoryScores.selfHarm) &&
        Objects.equals(this.selfHarmIntent, createModerationResponseResultsInnerCategoryScores.selfHarmIntent) &&
        Objects.equals(this.selfHarmInstructions, createModerationResponseResultsInnerCategoryScores.selfHarmInstructions) &&
        Objects.equals(this.sexual, createModerationResponseResultsInnerCategoryScores.sexual) &&
        Objects.equals(this.sexualMinors, createModerationResponseResultsInnerCategoryScores.sexualMinors) &&
        Objects.equals(this.violence, createModerationResponseResultsInnerCategoryScores.violence) &&
        Objects.equals(this.violenceGraphic, createModerationResponseResultsInnerCategoryScores.violenceGraphic);
  }

  @Override
  public int hashCode() {
    return Objects.hash(hate, hateThreatening, harassment, harassmentThreatening, selfHarm, selfHarmIntent, selfHarmInstructions, sexual, sexualMinors, violence, violenceGraphic);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateModerationResponseResultsInnerCategoryScores {\n");
    
    sb.append("    hate: ").append(toIndentedString(hate)).append("\n");
    sb.append("    hateThreatening: ").append(toIndentedString(hateThreatening)).append("\n");
    sb.append("    harassment: ").append(toIndentedString(harassment)).append("\n");
    sb.append("    harassmentThreatening: ").append(toIndentedString(harassmentThreatening)).append("\n");
    sb.append("    selfHarm: ").append(toIndentedString(selfHarm)).append("\n");
    sb.append("    selfHarmIntent: ").append(toIndentedString(selfHarmIntent)).append("\n");
    sb.append("    selfHarmInstructions: ").append(toIndentedString(selfHarmInstructions)).append("\n");
    sb.append("    sexual: ").append(toIndentedString(sexual)).append("\n");
    sb.append("    sexualMinors: ").append(toIndentedString(sexualMinors)).append("\n");
    sb.append("    violence: ").append(toIndentedString(violence)).append("\n");
    sb.append("    violenceGraphic: ").append(toIndentedString(violenceGraphic)).append("\n");
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

