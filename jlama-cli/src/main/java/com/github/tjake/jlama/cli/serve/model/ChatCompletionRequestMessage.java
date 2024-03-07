package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.ChatCompletionMessageToolCall;
import com.github.tjake.jlama.cli.serve.model.ChatCompletionRequestAssistantMessage;
import com.github.tjake.jlama.cli.serve.model.ChatCompletionRequestAssistantMessageFunctionCall;
import com.github.tjake.jlama.cli.serve.model.ChatCompletionRequestFunctionMessage;
import com.github.tjake.jlama.cli.serve.model.ChatCompletionRequestSystemMessage;
import com.github.tjake.jlama.cli.serve.model.ChatCompletionRequestToolMessage;
import com.github.tjake.jlama.cli.serve.model.ChatCompletionRequestUserMessage;
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



@JsonTypeName("ChatCompletionRequestMessage")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class ChatCompletionRequestMessage   {
  private @Valid String content;
  public enum RoleEnum {

    FUNCTION(String.valueOf("function"));


    private String value;

    RoleEnum (String v) {
        value = v;
    }

    public String value() {
        return value;
    }

    @Override
    @JsonValue
    public String toString() {
        return String.valueOf(value);
    }

    /**
     * Convert a String into String, as specified in the
     * <a href="https://download.oracle.com/otndocs/jcp/jaxrs-2_0-fr-eval-spec/index.html">See JAX RS 2.0 Specification, section 3.2, p. 12</a>
     */
    public static RoleEnum fromString(String s) {
        for (RoleEnum b : RoleEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected string value '" + s + "'");
    }

    @JsonCreator
    public static RoleEnum fromValue(String value) {
        for (RoleEnum b : RoleEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }
}

  private @Valid ChatCompletionRole role;
  private @Valid List<@Valid ChatCompletionMessageToolCall> toolCalls;
  private @Valid ChatCompletionRequestAssistantMessageFunctionCall functionCall;
  private @Valid String toolCallId;
  private @Valid String name;

  /**
   * The return value from the function call, to return to the model.
   **/
  public ChatCompletionRequestMessage content(String content) {
    this.content = content;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The return value from the function call, to return to the model.")
  @JsonProperty("content")
  @NotNull
  public String getContent() {
    return content;
  }

  @JsonProperty("content")
  public void setContent(String content) {
    this.content = content;
  }

  /**
   * The role of the messages author, in this case &#x60;function&#x60;.
   **/
  public ChatCompletionRequestMessage role(ChatCompletionRole role) {
    this.role = role;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The role of the messages author, in this case `function`.")
  @JsonProperty("role")
  @NotNull
  public ChatCompletionRole getRole() {
    return role;
  }

  @JsonProperty("role")
  public void setRole(ChatCompletionRole role) {
    this.role = role;
  }

  /**
   * The tool calls generated by the model, such as function calls.
   **/
  public ChatCompletionRequestMessage toolCalls(List<@Valid ChatCompletionMessageToolCall> toolCalls) {
    this.toolCalls = toolCalls;
    return this;
  }

  
  @ApiModelProperty(value = "The tool calls generated by the model, such as function calls.")
  @JsonProperty("tool_calls")
  public List<ChatCompletionMessageToolCall> getToolCalls() {
    return toolCalls;
  }

  @JsonProperty("tool_calls")
  public void setToolCalls(List<@Valid ChatCompletionMessageToolCall> toolCalls) {
    this.toolCalls = toolCalls;
  }

  public ChatCompletionRequestMessage addToolCallsItem(ChatCompletionMessageToolCall toolCallsItem) {
    if (this.toolCalls == null) {
      this.toolCalls = new ArrayList<>();
    }

    this.toolCalls.add(toolCallsItem);
    return this;
  }

  public ChatCompletionRequestMessage removeToolCallsItem(ChatCompletionMessageToolCall toolCallsItem) {
    if (toolCallsItem != null && this.toolCalls != null) {
      this.toolCalls.remove(toolCallsItem);
    }

    return this;
  }
  /**
   **/
  public ChatCompletionRequestMessage functionCall(ChatCompletionRequestAssistantMessageFunctionCall functionCall) {
    this.functionCall = functionCall;
    return this;
  }

  
  @ApiModelProperty(value = "")
  @JsonProperty("function_call")
  public ChatCompletionRequestAssistantMessageFunctionCall getFunctionCall() {
    return functionCall;
  }

  @JsonProperty("function_call")
  public void setFunctionCall(ChatCompletionRequestAssistantMessageFunctionCall functionCall) {
    this.functionCall = functionCall;
  }

  /**
   * Tool call that this message is responding to.
   **/
  public ChatCompletionRequestMessage toolCallId(String toolCallId) {
    this.toolCallId = toolCallId;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "Tool call that this message is responding to.")
  @JsonProperty("tool_call_id")
  @NotNull
  public String getToolCallId() {
    return toolCallId;
  }

  @JsonProperty("tool_call_id")
  public void setToolCallId(String toolCallId) {
    this.toolCallId = toolCallId;
  }

  /**
   * The name of the function to call.
   **/
  public ChatCompletionRequestMessage name(String name) {
    this.name = name;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The name of the function to call.")
  @JsonProperty("name")
  @NotNull
  public String getName() {
    return name;
  }

  @JsonProperty("name")
  public void setName(String name) {
    this.name = name;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    ChatCompletionRequestMessage chatCompletionRequestMessage = (ChatCompletionRequestMessage) o;
    return Objects.equals(this.content, chatCompletionRequestMessage.content) &&
        Objects.equals(this.role, chatCompletionRequestMessage.role) &&
        Objects.equals(this.toolCalls, chatCompletionRequestMessage.toolCalls) &&
        Objects.equals(this.functionCall, chatCompletionRequestMessage.functionCall) &&
        Objects.equals(this.toolCallId, chatCompletionRequestMessage.toolCallId) &&
        Objects.equals(this.name, chatCompletionRequestMessage.name);
  }

  @Override
  public int hashCode() {
    return Objects.hash(content, role, toolCalls, functionCall, toolCallId, name);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class ChatCompletionRequestMessage {\n");
    
    sb.append("    content: ").append(toIndentedString(content)).append("\n");
    sb.append("    role: ").append(toIndentedString(role)).append("\n");
    sb.append("    toolCalls: ").append(toIndentedString(toolCalls)).append("\n");
    sb.append("    functionCall: ").append(toIndentedString(functionCall)).append("\n");
    sb.append("    toolCallId: ").append(toIndentedString(toolCallId)).append("\n");
    sb.append("    name: ").append(toIndentedString(name)).append("\n");
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

