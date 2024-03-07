package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CreateCompletionRequestModel;
import com.github.tjake.jlama.cli.serve.model.CreateCompletionRequestPrompt;
import com.github.tjake.jlama.cli.serve.model.CreateCompletionRequestStop;
import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

import jakarta.validation.constraints.*;
import jakarta.validation.Valid;

import io.swagger.annotations.*;
import java.util.Objects;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import com.fasterxml.jackson.annotation.JsonTypeName;



@JsonTypeName("CreateCompletionRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateCompletionRequest   {
  private @Valid CreateCompletionRequestModel model;
  private @Valid String prompt = "<|endoftext|>";
  private @Valid Integer bestOf = 1;
  private @Valid Boolean echo = false;
  private @Valid BigDecimal frequencyPenalty = new BigDecimal("0");
  private @Valid Map<String, Integer> logitBias;
  private @Valid Integer logprobs;
  private @Valid Integer maxTokens = 16;
  private @Valid Integer n = 1;
  private @Valid BigDecimal presencePenalty = new BigDecimal("0");
  private @Valid Integer seed;
  private @Valid CreateCompletionRequestStop stop = null;
  private @Valid Boolean stream = false;
  private @Valid String suffix;
  private @Valid BigDecimal temperature = new BigDecimal("1");
  private @Valid BigDecimal topP = new BigDecimal("1");
  private @Valid String user;

  /**
   **/
  public CreateCompletionRequest model(CreateCompletionRequestModel model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("model")
  @NotNull
  public CreateCompletionRequestModel getModel() {
    return model;
  }

  @JsonProperty("model")
  public void setModel(CreateCompletionRequestModel model) {
    this.model = model;
  }

  /**
   **/
  public CreateCompletionRequest prompt(String prompt) {
    this.prompt = prompt;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("prompt")
  @NotNull
  public String getPrompt() {
    return prompt;
  }

  @JsonProperty("prompt")
  public void setPrompt(String prompt) {
    this.prompt = prompt;
  }

  /**
   * Generates &#x60;best_of&#x60; completions server-side and returns the \&quot;best\&quot; (the one with the highest log probability per token). Results cannot be streamed.  When used with &#x60;n&#x60;, &#x60;best_of&#x60; controls the number of candidate completions and &#x60;n&#x60; specifies how many to return – &#x60;best_of&#x60; must be greater than &#x60;n&#x60;.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for &#x60;max_tokens&#x60; and &#x60;stop&#x60;. 
   * minimum: 0
   * maximum: 20
   **/
  public CreateCompletionRequest bestOf(Integer bestOf) {
    this.bestOf = bestOf;
    return this;
  }

  
  @ApiModelProperty(value = "Generates `best_of` completions server-side and returns the \"best\" (the one with the highest log probability per token). Results cannot be streamed.  When used with `n`, `best_of` controls the number of candidate completions and `n` specifies how many to return – `best_of` must be greater than `n`.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`. ")
  @JsonProperty("best_of")
 @Min(0) @Max(20)  public Integer getBestOf() {
    return bestOf;
  }

  @JsonProperty("best_of")
  public void setBestOf(Integer bestOf) {
    this.bestOf = bestOf;
  }

  /**
   * Echo back the prompt in addition to the completion 
   **/
  public CreateCompletionRequest echo(Boolean echo) {
    this.echo = echo;
    return this;
  }

  
  @ApiModelProperty(value = "Echo back the prompt in addition to the completion ")
  @JsonProperty("echo")
  public Boolean getEcho() {
    return echo;
  }

  @JsonProperty("echo")
  public void setEcho(Boolean echo) {
    this.echo = echo;
  }

  /**
   * Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model&#39;s likelihood to repeat the same line verbatim.  [See more information about frequency and presence penalties.](/docs/guides/gpt/parameter-details) 
   * minimum: -2
   * maximum: 2
   **/
  public CreateCompletionRequest frequencyPenalty(BigDecimal frequencyPenalty) {
    this.frequencyPenalty = frequencyPenalty;
    return this;
  }

  
  @ApiModelProperty(value = "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.  [See more information about frequency and presence penalties.](/docs/guides/gpt/parameter-details) ")
  @JsonProperty("frequency_penalty")
 @DecimalMin("-2") @DecimalMax("2")  public BigDecimal getFrequencyPenalty() {
    return frequencyPenalty;
  }

  @JsonProperty("frequency_penalty")
  public void setFrequencyPenalty(BigDecimal frequencyPenalty) {
    this.frequencyPenalty = frequencyPenalty;
  }

  /**
   * Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](/tokenizer?view&#x3D;bpe) (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  As an example, you can pass &#x60;{\&quot;50256\&quot;: -100}&#x60; to prevent the &lt;|endoftext|&gt; token from being generated. 
   **/
  public CreateCompletionRequest logitBias(Map<String, Integer> logitBias) {
    this.logitBias = logitBias;
    return this;
  }

  
  @ApiModelProperty(value = "Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  As an example, you can pass `{\"50256\": -100}` to prevent the <|endoftext|> token from being generated. ")
  @JsonProperty("logit_bias")
  public Map<String, Integer> getLogitBias() {
    return logitBias;
  }

  @JsonProperty("logit_bias")
  public void setLogitBias(Map<String, Integer> logitBias) {
    this.logitBias = logitBias;
  }

  public CreateCompletionRequest putLogitBiasItem(String key, Integer logitBiasItem) {
    if (this.logitBias == null) {
      this.logitBias = new HashMap<>();
    }

    this.logitBias.put(key, logitBiasItem);
    return this;
  }

  public CreateCompletionRequest removeLogitBiasItem(Integer logitBiasItem) {
    if (logitBiasItem != null && this.logitBias != null) {
      this.logitBias.remove(logitBiasItem);
    }

    return this;
  }
  /**
   * Include the log probabilities on the &#x60;logprobs&#x60; most likely tokens, as well the chosen tokens. For example, if &#x60;logprobs&#x60; is 5, the API will return a list of the 5 most likely tokens. The API will always return the &#x60;logprob&#x60; of the sampled token, so there may be up to &#x60;logprobs+1&#x60; elements in the response.  The maximum value for &#x60;logprobs&#x60; is 5. 
   * minimum: 0
   * maximum: 5
   **/
  public CreateCompletionRequest logprobs(Integer logprobs) {
    this.logprobs = logprobs;
    return this;
  }

  
  @ApiModelProperty(value = "Include the log probabilities on the `logprobs` most likely tokens, as well the chosen tokens. For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens. The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response.  The maximum value for `logprobs` is 5. ")
  @JsonProperty("logprobs")
 @Min(0) @Max(5)  public Integer getLogprobs() {
    return logprobs;
  }

  @JsonProperty("logprobs")
  public void setLogprobs(Integer logprobs) {
    this.logprobs = logprobs;
  }

  /**
   * The maximum number of [tokens](/tokenizer) to generate in the completion.  The token count of your prompt plus &#x60;max_tokens&#x60; cannot exceed the model&#39;s context length. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens. 
   * minimum: 0
   **/
  public CreateCompletionRequest maxTokens(Integer maxTokens) {
    this.maxTokens = maxTokens;
    return this;
  }

  
  @ApiModelProperty(example = "16", value = "The maximum number of [tokens](/tokenizer) to generate in the completion.  The token count of your prompt plus `max_tokens` cannot exceed the model's context length. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens. ")
  @JsonProperty("max_tokens")
 @Min(0)  public Integer getMaxTokens() {
    return maxTokens;
  }

  @JsonProperty("max_tokens")
  public void setMaxTokens(Integer maxTokens) {
    this.maxTokens = maxTokens;
  }

  /**
   * How many completions to generate for each prompt.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for &#x60;max_tokens&#x60; and &#x60;stop&#x60;. 
   * minimum: 1
   * maximum: 128
   **/
  public CreateCompletionRequest n(Integer n) {
    this.n = n;
    return this;
  }

  
  @ApiModelProperty(example = "1", value = "How many completions to generate for each prompt.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`. ")
  @JsonProperty("n")
 @Min(1) @Max(128)  public Integer getN() {
    return n;
  }

  @JsonProperty("n")
  public void setN(Integer n) {
    this.n = n;
  }

  /**
   * Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model&#39;s likelihood to talk about new topics.  [See more information about frequency and presence penalties.](/docs/guides/gpt/parameter-details) 
   * minimum: -2
   * maximum: 2
   **/
  public CreateCompletionRequest presencePenalty(BigDecimal presencePenalty) {
    this.presencePenalty = presencePenalty;
    return this;
  }

  
  @ApiModelProperty(value = "Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.  [See more information about frequency and presence penalties.](/docs/guides/gpt/parameter-details) ")
  @JsonProperty("presence_penalty")
 @DecimalMin("-2") @DecimalMax("2")  public BigDecimal getPresencePenalty() {
    return presencePenalty;
  }

  @JsonProperty("presence_penalty")
  public void setPresencePenalty(BigDecimal presencePenalty) {
    this.presencePenalty = presencePenalty;
  }

  /**
   * If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same &#x60;seed&#x60; and parameters should return the same result.  Determinism is not guaranteed, and you should refer to the &#x60;system_fingerprint&#x60; response parameter to monitor changes in the backend. 
   * minimum: -9223372036854775808
   * maximum: 9223372036854775807
   **/
  public CreateCompletionRequest seed(Integer seed) {
    this.seed = seed;
    return this;
  }

  
  @ApiModelProperty(value = "If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result.  Determinism is not guaranteed, and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend. ")
  @JsonProperty("seed")
 @Min(Integer.MIN_VALUE) @Max(Integer.MAX_VALUE)  public Integer getSeed() {
    return seed;
  }

  @JsonProperty("seed")
  public void setSeed(Integer seed) {
    this.seed = seed;
  }

  /**
   **/
  public CreateCompletionRequest stop(CreateCompletionRequestStop stop) {
    this.stop = stop;
    return this;
  }

  
  @ApiModelProperty(value = "")
  @JsonProperty("stop")
  public CreateCompletionRequestStop getStop() {
    return stop;
  }

  @JsonProperty("stop")
  public void setStop(CreateCompletionRequestStop stop) {
    this.stop = stop;
  }

  /**
   * Whether to stream back partial progress. If set, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a &#x60;data: [DONE]&#x60; message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions). 
   **/
  public CreateCompletionRequest stream(Boolean stream) {
    this.stream = stream;
    return this;
  }

  
  @ApiModelProperty(value = "Whether to stream back partial progress. If set, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message. [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions). ")
  @JsonProperty("stream")
  public Boolean getStream() {
    return stream;
  }

  @JsonProperty("stream")
  public void setStream(Boolean stream) {
    this.stream = stream;
  }

  /**
   * The suffix that comes after a completion of inserted text.
   **/
  public CreateCompletionRequest suffix(String suffix) {
    this.suffix = suffix;
    return this;
  }

  
  @ApiModelProperty(example = "test.", value = "The suffix that comes after a completion of inserted text.")
  @JsonProperty("suffix")
  public String getSuffix() {
    return suffix;
  }

  @JsonProperty("suffix")
  public void setSuffix(String suffix) {
    this.suffix = suffix;
  }

  /**
   * What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or &#x60;top_p&#x60; but not both. 
   * minimum: 0
   * maximum: 2
   **/
  public CreateCompletionRequest temperature(BigDecimal temperature) {
    this.temperature = temperature;
    return this;
  }

  
  @ApiModelProperty(example = "1", value = "What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or `top_p` but not both. ")
  @JsonProperty("temperature")
 @DecimalMin("0") @DecimalMax("2")  public BigDecimal getTemperature() {
    return temperature;
  }

  @JsonProperty("temperature")
  public void setTemperature(BigDecimal temperature) {
    this.temperature = temperature;
  }

  /**
   * An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or &#x60;temperature&#x60; but not both. 
   * minimum: 0
   * maximum: 1
   **/
  public CreateCompletionRequest topP(BigDecimal topP) {
    this.topP = topP;
    return this;
  }

  
  @ApiModelProperty(example = "1", value = "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or `temperature` but not both. ")
  @JsonProperty("top_p")
 @DecimalMin("0") @DecimalMax("1")  public BigDecimal getTopP() {
    return topP;
  }

  @JsonProperty("top_p")
  public void setTopP(BigDecimal topP) {
    this.topP = topP;
  }

  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). 
   **/
  public CreateCompletionRequest user(String user) {
    this.user = user;
    return this;
  }

  
  @ApiModelProperty(example = "user-1234", value = "A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices/end-user-ids). ")
  @JsonProperty("user")
  public String getUser() {
    return user;
  }

  @JsonProperty("user")
  public void setUser(String user) {
    this.user = user;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateCompletionRequest createCompletionRequest = (CreateCompletionRequest) o;
    return Objects.equals(this.model, createCompletionRequest.model) &&
        Objects.equals(this.prompt, createCompletionRequest.prompt) &&
        Objects.equals(this.bestOf, createCompletionRequest.bestOf) &&
        Objects.equals(this.echo, createCompletionRequest.echo) &&
        Objects.equals(this.frequencyPenalty, createCompletionRequest.frequencyPenalty) &&
        Objects.equals(this.logitBias, createCompletionRequest.logitBias) &&
        Objects.equals(this.logprobs, createCompletionRequest.logprobs) &&
        Objects.equals(this.maxTokens, createCompletionRequest.maxTokens) &&
        Objects.equals(this.n, createCompletionRequest.n) &&
        Objects.equals(this.presencePenalty, createCompletionRequest.presencePenalty) &&
        Objects.equals(this.seed, createCompletionRequest.seed) &&
        Objects.equals(this.stop, createCompletionRequest.stop) &&
        Objects.equals(this.stream, createCompletionRequest.stream) &&
        Objects.equals(this.suffix, createCompletionRequest.suffix) &&
        Objects.equals(this.temperature, createCompletionRequest.temperature) &&
        Objects.equals(this.topP, createCompletionRequest.topP) &&
        Objects.equals(this.user, createCompletionRequest.user);
  }

  @Override
  public int hashCode() {
    return Objects.hash(model, prompt, bestOf, echo, frequencyPenalty, logitBias, logprobs, maxTokens, n, presencePenalty, seed, stop, stream, suffix, temperature, topP, user);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateCompletionRequest {\n");
    
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    prompt: ").append(toIndentedString(prompt)).append("\n");
    sb.append("    bestOf: ").append(toIndentedString(bestOf)).append("\n");
    sb.append("    echo: ").append(toIndentedString(echo)).append("\n");
    sb.append("    frequencyPenalty: ").append(toIndentedString(frequencyPenalty)).append("\n");
    sb.append("    logitBias: ").append(toIndentedString(logitBias)).append("\n");
    sb.append("    logprobs: ").append(toIndentedString(logprobs)).append("\n");
    sb.append("    maxTokens: ").append(toIndentedString(maxTokens)).append("\n");
    sb.append("    n: ").append(toIndentedString(n)).append("\n");
    sb.append("    presencePenalty: ").append(toIndentedString(presencePenalty)).append("\n");
    sb.append("    seed: ").append(toIndentedString(seed)).append("\n");
    sb.append("    stop: ").append(toIndentedString(stop)).append("\n");
    sb.append("    stream: ").append(toIndentedString(stream)).append("\n");
    sb.append("    suffix: ").append(toIndentedString(suffix)).append("\n");
    sb.append("    temperature: ").append(toIndentedString(temperature)).append("\n");
    sb.append("    topP: ").append(toIndentedString(topP)).append("\n");
    sb.append("    user: ").append(toIndentedString(user)).append("\n");
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

