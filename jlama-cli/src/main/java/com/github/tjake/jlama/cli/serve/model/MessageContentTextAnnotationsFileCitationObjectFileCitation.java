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



@JsonTypeName("MessageContentTextAnnotationsFileCitationObject_file_citation")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class MessageContentTextAnnotationsFileCitationObjectFileCitation   {
  private @Valid String fileId;
  private @Valid String quote;

  /**
   * The ID of the specific File the citation is from.
   **/
  public MessageContentTextAnnotationsFileCitationObjectFileCitation fileId(String fileId) {
    this.fileId = fileId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The ID of the specific File the citation is from.")
  @JsonProperty("file_id")
  @NotNull
  public String getFileId() {
    return fileId;
  }

  @JsonProperty("file_id")
  public void setFileId(String fileId) {
    this.fileId = fileId;
  }

  /**
   * The specific quote in the file.
   **/
  public MessageContentTextAnnotationsFileCitationObjectFileCitation quote(String quote) {
    this.quote = quote;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The specific quote in the file.")
  @JsonProperty("quote")
  @NotNull
  public String getQuote() {
    return quote;
  }

  @JsonProperty("quote")
  public void setQuote(String quote) {
    this.quote = quote;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    MessageContentTextAnnotationsFileCitationObjectFileCitation messageContentTextAnnotationsFileCitationObjectFileCitation = (MessageContentTextAnnotationsFileCitationObjectFileCitation) o;
    return Objects.equals(this.fileId, messageContentTextAnnotationsFileCitationObjectFileCitation.fileId) &&
        Objects.equals(this.quote, messageContentTextAnnotationsFileCitationObjectFileCitation.quote);
  }

  @Override
  public int hashCode() {
    return Objects.hash(fileId, quote);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class MessageContentTextAnnotationsFileCitationObjectFileCitation {\n");
    
    sb.append("    fileId: ").append(toIndentedString(fileId)).append("\n");
    sb.append("    quote: ").append(toIndentedString(quote)).append("\n");
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

