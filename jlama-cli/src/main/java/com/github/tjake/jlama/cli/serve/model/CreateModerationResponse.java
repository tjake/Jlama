package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CreateModerationResponseResultsInner;
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
 * Represents policy compliance report by OpenAI&#39;s content moderation model against a given input.
 **/
@ApiModel(description = "Represents policy compliance report by OpenAI's content moderation model against a given input.")
@JsonTypeName("CreateModerationResponse")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateModerationResponse   {
  private @Valid String id;
  private @Valid String model;
  private @Valid List<@Valid CreateModerationResponseResultsInner> results = new ArrayList<>();

  /**
   * The unique identifier for the moderation request.
   **/
  public CreateModerationResponse id(String id) {
    this.id = id;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The unique identifier for the moderation request.")
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
   * The model used to generate the moderation results.
   **/
  public CreateModerationResponse model(String model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The model used to generate the moderation results.")
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
   * A list of moderation objects.
   **/
  public CreateModerationResponse results(List<@Valid CreateModerationResponseResultsInner> results) {
    this.results = results;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "A list of moderation objects.")
  @JsonProperty("results")
  @NotNull
  public List<CreateModerationResponseResultsInner> getResults() {
    return results;
  }

  @JsonProperty("results")
  public void setResults(List<@Valid CreateModerationResponseResultsInner> results) {
    this.results = results;
  }

  public CreateModerationResponse addResultsItem(CreateModerationResponseResultsInner resultsItem) {
    if (this.results == null) {
      this.results = new ArrayList<>();
    }

    this.results.add(resultsItem);
    return this;
  }

  public CreateModerationResponse removeResultsItem(CreateModerationResponseResultsInner resultsItem) {
    if (resultsItem != null && this.results != null) {
      this.results.remove(resultsItem);
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
    CreateModerationResponse createModerationResponse = (CreateModerationResponse) o;
    return Objects.equals(this.id, createModerationResponse.id) &&
        Objects.equals(this.model, createModerationResponse.model) &&
        Objects.equals(this.results, createModerationResponse.results);
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, model, results);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateModerationResponse {\n");
    
    sb.append("    id: ").append(toIndentedString(id)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    results: ").append(toIndentedString(results)).append("\n");
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

