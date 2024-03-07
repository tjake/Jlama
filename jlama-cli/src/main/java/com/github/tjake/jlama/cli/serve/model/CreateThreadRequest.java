package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CreateMessageRequest;
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



@JsonTypeName("CreateThreadRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateThreadRequest   {
  private @Valid List<@Valid CreateMessageRequest> messages;
  private @Valid Object metadata;

  /**
   * A list of [messages](/docs/api-reference/messages) to start the thread with.
   **/
  public CreateThreadRequest messages(List<@Valid CreateMessageRequest> messages) {
    this.messages = messages;
    return this;
  }

  
  @ApiModelProperty(value = "A list of [messages](/docs/api-reference/messages) to start the thread with.")
  @JsonProperty("messages")
  public List<CreateMessageRequest> getMessages() {
    return messages;
  }

  @JsonProperty("messages")
  public void setMessages(List<@Valid CreateMessageRequest> messages) {
    this.messages = messages;
  }

  public CreateThreadRequest addMessagesItem(CreateMessageRequest messagesItem) {
    if (this.messages == null) {
      this.messages = new ArrayList<>();
    }

    this.messages.add(messagesItem);
    return this;
  }

  public CreateThreadRequest removeMessagesItem(CreateMessageRequest messagesItem) {
    if (messagesItem != null && this.messages != null) {
      this.messages.remove(messagesItem);
    }

    return this;
  }
  /**
   * metadata_description
   **/
  public CreateThreadRequest metadata(Object metadata) {
    this.metadata = metadata;
    return this;
  }

  
  @ApiModelProperty(value = "metadata_description")
  @JsonProperty("metadata")
  public Object getMetadata() {
    return metadata;
  }

  @JsonProperty("metadata")
  public void setMetadata(Object metadata) {
    this.metadata = metadata;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateThreadRequest createThreadRequest = (CreateThreadRequest) o;
    return Objects.equals(this.messages, createThreadRequest.messages) &&
        Objects.equals(this.metadata, createThreadRequest.metadata);
  }

  @Override
  public int hashCode() {
    return Objects.hash(messages, metadata);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateThreadRequest {\n");
    
    sb.append("    messages: ").append(toIndentedString(messages)).append("\n");
    sb.append("    metadata: ").append(toIndentedString(metadata)).append("\n");
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

