package com.github.tjake.jlama.cli.serve.model;

import com.fasterxml.jackson.annotation.JsonTypeName;
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
 * A list of the categories, and whether they are flagged or not.
 **/
@ApiModel(description = "A list of the categories, and whether they are flagged or not.")
@JsonTypeName("CreateModerationResponse_results_inner_categories")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateModerationResponseResultsInnerCategories   {
  private @Valid Boolean hate;
  private @Valid Boolean hateThreatening;
  private @Valid Boolean harassment;
  private @Valid Boolean harassmentThreatening;
  private @Valid Boolean selfHarm;
  private @Valid Boolean selfHarmIntent;
  private @Valid Boolean selfHarmInstructions;
  private @Valid Boolean sexual;
  private @Valid Boolean sexualMinors;
  private @Valid Boolean violence;
  private @Valid Boolean violenceGraphic;

  /**
   * Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non-protected groups (e.g., chess players) is harrassment.
   **/
  public CreateModerationResponseResultsInnerCategories hate(Boolean hate) {
    this.hate = hate;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Hateful content aimed at non-protected groups (e.g., chess players) is harrassment.")
  @JsonProperty("hate")
  @NotNull
  public Boolean getHate() {
    return hate;
  }

  @JsonProperty("hate")
  public void setHate(Boolean hate) {
    this.hate = hate;
  }

  /**
   * Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.
   **/
  public CreateModerationResponseResultsInnerCategories hateThreatening(Boolean hateThreatening) {
    this.hateThreatening = hateThreatening;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Hateful content that also includes violence or serious harm towards the targeted group based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.")
  @JsonProperty("hate/threatening")
  @NotNull
  public Boolean getHateThreatening() {
    return hateThreatening;
  }

  @JsonProperty("hate/threatening")
  public void setHateThreatening(Boolean hateThreatening) {
    this.hateThreatening = hateThreatening;
  }

  /**
   * Content that expresses, incites, or promotes harassing language towards any target.
   **/
  public CreateModerationResponseResultsInnerCategories harassment(Boolean harassment) {
    this.harassment = harassment;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Content that expresses, incites, or promotes harassing language towards any target.")
  @JsonProperty("harassment")
  @NotNull
  public Boolean getHarassment() {
    return harassment;
  }

  @JsonProperty("harassment")
  public void setHarassment(Boolean harassment) {
    this.harassment = harassment;
  }

  /**
   * Harassment content that also includes violence or serious harm towards any target.
   **/
  public CreateModerationResponseResultsInnerCategories harassmentThreatening(Boolean harassmentThreatening) {
    this.harassmentThreatening = harassmentThreatening;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Harassment content that also includes violence or serious harm towards any target.")
  @JsonProperty("harassment/threatening")
  @NotNull
  public Boolean getHarassmentThreatening() {
    return harassmentThreatening;
  }

  @JsonProperty("harassment/threatening")
  public void setHarassmentThreatening(Boolean harassmentThreatening) {
    this.harassmentThreatening = harassmentThreatening;
  }

  /**
   * Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.
   **/
  public CreateModerationResponseResultsInnerCategories selfHarm(Boolean selfHarm) {
    this.selfHarm = selfHarm;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.")
  @JsonProperty("self-harm")
  @NotNull
  public Boolean getSelfHarm() {
    return selfHarm;
  }

  @JsonProperty("self-harm")
  public void setSelfHarm(Boolean selfHarm) {
    this.selfHarm = selfHarm;
  }

  /**
   * Content where the speaker expresses that they are engaging or intend to engage in acts of self-harm, such as suicide, cutting, and eating disorders.
   **/
  public CreateModerationResponseResultsInnerCategories selfHarmIntent(Boolean selfHarmIntent) {
    this.selfHarmIntent = selfHarmIntent;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Content where the speaker expresses that they are engaging or intend to engage in acts of self-harm, such as suicide, cutting, and eating disorders.")
  @JsonProperty("self-harm/intent")
  @NotNull
  public Boolean getSelfHarmIntent() {
    return selfHarmIntent;
  }

  @JsonProperty("self-harm/intent")
  public void setSelfHarmIntent(Boolean selfHarmIntent) {
    this.selfHarmIntent = selfHarmIntent;
  }

  /**
   * Content that encourages performing acts of self-harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts.
   **/
  public CreateModerationResponseResultsInnerCategories selfHarmInstructions(Boolean selfHarmInstructions) {
    this.selfHarmInstructions = selfHarmInstructions;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Content that encourages performing acts of self-harm, such as suicide, cutting, and eating disorders, or that gives instructions or advice on how to commit such acts.")
  @JsonProperty("self-harm/instructions")
  @NotNull
  public Boolean getSelfHarmInstructions() {
    return selfHarmInstructions;
  }

  @JsonProperty("self-harm/instructions")
  public void setSelfHarmInstructions(Boolean selfHarmInstructions) {
    this.selfHarmInstructions = selfHarmInstructions;
  }

  /**
   * Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).
   **/
  public CreateModerationResponseResultsInnerCategories sexual(Boolean sexual) {
    this.sexual = sexual;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).")
  @JsonProperty("sexual")
  @NotNull
  public Boolean getSexual() {
    return sexual;
  }

  @JsonProperty("sexual")
  public void setSexual(Boolean sexual) {
    this.sexual = sexual;
  }

  /**
   * Sexual content that includes an individual who is under 18 years old.
   **/
  public CreateModerationResponseResultsInnerCategories sexualMinors(Boolean sexualMinors) {
    this.sexualMinors = sexualMinors;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Sexual content that includes an individual who is under 18 years old.")
  @JsonProperty("sexual/minors")
  @NotNull
  public Boolean getSexualMinors() {
    return sexualMinors;
  }

  @JsonProperty("sexual/minors")
  public void setSexualMinors(Boolean sexualMinors) {
    this.sexualMinors = sexualMinors;
  }

  /**
   * Content that depicts death, violence, or physical injury.
   **/
  public CreateModerationResponseResultsInnerCategories violence(Boolean violence) {
    this.violence = violence;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Content that depicts death, violence, or physical injury.")
  @JsonProperty("violence")
  @NotNull
  public Boolean getViolence() {
    return violence;
  }

  @JsonProperty("violence")
  public void setViolence(Boolean violence) {
    this.violence = violence;
  }

  /**
   * Content that depicts death, violence, or physical injury in graphic detail.
   **/
  public CreateModerationResponseResultsInnerCategories violenceGraphic(Boolean violenceGraphic) {
    this.violenceGraphic = violenceGraphic;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Content that depicts death, violence, or physical injury in graphic detail.")
  @JsonProperty("violence/graphic")
  @NotNull
  public Boolean getViolenceGraphic() {
    return violenceGraphic;
  }

  @JsonProperty("violence/graphic")
  public void setViolenceGraphic(Boolean violenceGraphic) {
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
    CreateModerationResponseResultsInnerCategories createModerationResponseResultsInnerCategories = (CreateModerationResponseResultsInnerCategories) o;
    return Objects.equals(this.hate, createModerationResponseResultsInnerCategories.hate) &&
        Objects.equals(this.hateThreatening, createModerationResponseResultsInnerCategories.hateThreatening) &&
        Objects.equals(this.harassment, createModerationResponseResultsInnerCategories.harassment) &&
        Objects.equals(this.harassmentThreatening, createModerationResponseResultsInnerCategories.harassmentThreatening) &&
        Objects.equals(this.selfHarm, createModerationResponseResultsInnerCategories.selfHarm) &&
        Objects.equals(this.selfHarmIntent, createModerationResponseResultsInnerCategories.selfHarmIntent) &&
        Objects.equals(this.selfHarmInstructions, createModerationResponseResultsInnerCategories.selfHarmInstructions) &&
        Objects.equals(this.sexual, createModerationResponseResultsInnerCategories.sexual) &&
        Objects.equals(this.sexualMinors, createModerationResponseResultsInnerCategories.sexualMinors) &&
        Objects.equals(this.violence, createModerationResponseResultsInnerCategories.violence) &&
        Objects.equals(this.violenceGraphic, createModerationResponseResultsInnerCategories.violenceGraphic);
  }

  @Override
  public int hashCode() {
    return Objects.hash(hate, hateThreatening, harassment, harassmentThreatening, selfHarm, selfHarmIntent, selfHarmInstructions, sexual, sexualMinors, violence, violenceGraphic);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateModerationResponseResultsInnerCategories {\n");
    
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

