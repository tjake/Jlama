package com.github.tjake.jlama.cli.serve.model;

import com.fasterxml.jackson.annotation.JsonTypeName;
import com.github.tjake.jlama.cli.serve.model.MessageContentTextObjectTextAnnotationsInner;
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



@JsonTypeName("MessageContentTextObject_text")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class MessageContentTextObjectText   {
  private @Valid String value;
  private @Valid List<@Valid MessageContentTextObjectTextAnnotationsInner> annotations = new ArrayList<>();

  /**
   * The data that makes up the text.
   **/
  public MessageContentTextObjectText value(String value) {
    this.value = value;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The data that makes up the text.")
  @JsonProperty("value")
  @NotNull
  public String getValue() {
    return value;
  }

  @JsonProperty("value")
  public void setValue(String value) {
    this.value = value;
  }

  /**
   **/
  public MessageContentTextObjectText annotations(List<@Valid MessageContentTextObjectTextAnnotationsInner> annotations) {
    this.annotations = annotations;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("annotations")
  @NotNull
  public List<MessageContentTextObjectTextAnnotationsInner> getAnnotations() {
    return annotations;
  }

  @JsonProperty("annotations")
  public void setAnnotations(List<@Valid MessageContentTextObjectTextAnnotationsInner> annotations) {
    this.annotations = annotations;
  }

  public MessageContentTextObjectText addAnnotationsItem(MessageContentTextObjectTextAnnotationsInner annotationsItem) {
    if (this.annotations == null) {
      this.annotations = new ArrayList<>();
    }

    this.annotations.add(annotationsItem);
    return this;
  }

  public MessageContentTextObjectText removeAnnotationsItem(MessageContentTextObjectTextAnnotationsInner annotationsItem) {
    if (annotationsItem != null && this.annotations != null) {
      this.annotations.remove(annotationsItem);
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
    MessageContentTextObjectText messageContentTextObjectText = (MessageContentTextObjectText) o;
    return Objects.equals(this.value, messageContentTextObjectText.value) &&
        Objects.equals(this.annotations, messageContentTextObjectText.annotations);
  }

  @Override
  public int hashCode() {
    return Objects.hash(value, annotations);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class MessageContentTextObjectText {\n");
    
    sb.append("    value: ").append(toIndentedString(value)).append("\n");
    sb.append("    annotations: ").append(toIndentedString(annotations)).append("\n");
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

