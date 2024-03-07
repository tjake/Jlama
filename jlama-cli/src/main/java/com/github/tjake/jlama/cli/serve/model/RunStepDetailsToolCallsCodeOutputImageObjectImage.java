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



@JsonTypeName("RunStepDetailsToolCallsCodeOutputImageObject_image")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class RunStepDetailsToolCallsCodeOutputImageObjectImage   {
  private @Valid String fileId;

  /**
   * The [file](/docs/api-reference/files) ID of the image.
   **/
  public RunStepDetailsToolCallsCodeOutputImageObjectImage fileId(String fileId) {
    this.fileId = fileId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The [file](/docs/api-reference/files) ID of the image.")
  @JsonProperty("file_id")
  @NotNull
  public String getFileId() {
    return fileId;
  }

  @JsonProperty("file_id")
  public void setFileId(String fileId) {
    this.fileId = fileId;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    RunStepDetailsToolCallsCodeOutputImageObjectImage runStepDetailsToolCallsCodeOutputImageObjectImage = (RunStepDetailsToolCallsCodeOutputImageObjectImage) o;
    return Objects.equals(this.fileId, runStepDetailsToolCallsCodeOutputImageObjectImage.fileId);
  }

  @Override
  public int hashCode() {
    return Objects.hash(fileId);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class RunStepDetailsToolCallsCodeOutputImageObjectImage {\n");
    
    sb.append("    fileId: ").append(toIndentedString(fileId)).append("\n");
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

