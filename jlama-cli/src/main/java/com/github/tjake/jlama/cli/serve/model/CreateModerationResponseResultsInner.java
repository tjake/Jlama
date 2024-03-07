package com.github.tjake.jlama.cli.serve.model;

import com.fasterxml.jackson.annotation.JsonTypeName;
import com.github.tjake.jlama.cli.serve.model.CreateModerationResponseResultsInnerCategories;
import com.github.tjake.jlama.cli.serve.model.CreateModerationResponseResultsInnerCategoryScores;
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



@JsonTypeName("CreateModerationResponse_results_inner")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateModerationResponseResultsInner   {
  private @Valid Boolean flagged;
  private @Valid CreateModerationResponseResultsInnerCategories categories;
  private @Valid CreateModerationResponseResultsInnerCategoryScores categoryScores;

  /**
   * Whether the content violates [OpenAI&#39;s usage policies](/policies/usage-policies).
   **/
  public CreateModerationResponseResultsInner flagged(Boolean flagged) {
    this.flagged = flagged;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Whether the content violates [OpenAI's usage policies](/policies/usage-policies).")
  @JsonProperty("flagged")
  @NotNull
  public Boolean getFlagged() {
    return flagged;
  }

  @JsonProperty("flagged")
  public void setFlagged(Boolean flagged) {
    this.flagged = flagged;
  }

  /**
   **/
  public CreateModerationResponseResultsInner categories(CreateModerationResponseResultsInnerCategories categories) {
    this.categories = categories;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("categories")
  @NotNull
  public CreateModerationResponseResultsInnerCategories getCategories() {
    return categories;
  }

  @JsonProperty("categories")
  public void setCategories(CreateModerationResponseResultsInnerCategories categories) {
    this.categories = categories;
  }

  /**
   **/
  public CreateModerationResponseResultsInner categoryScores(CreateModerationResponseResultsInnerCategoryScores categoryScores) {
    this.categoryScores = categoryScores;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("category_scores")
  @NotNull
  public CreateModerationResponseResultsInnerCategoryScores getCategoryScores() {
    return categoryScores;
  }

  @JsonProperty("category_scores")
  public void setCategoryScores(CreateModerationResponseResultsInnerCategoryScores categoryScores) {
    this.categoryScores = categoryScores;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateModerationResponseResultsInner createModerationResponseResultsInner = (CreateModerationResponseResultsInner) o;
    return Objects.equals(this.flagged, createModerationResponseResultsInner.flagged) &&
        Objects.equals(this.categories, createModerationResponseResultsInner.categories) &&
        Objects.equals(this.categoryScores, createModerationResponseResultsInner.categoryScores);
  }

  @Override
  public int hashCode() {
    return Objects.hash(flagged, categories, categoryScores);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateModerationResponseResultsInner {\n");
    
    sb.append("    flagged: ").append(toIndentedString(flagged)).append("\n");
    sb.append("    categories: ").append(toIndentedString(categories)).append("\n");
    sb.append("    categoryScores: ").append(toIndentedString(categoryScores)).append("\n");
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

