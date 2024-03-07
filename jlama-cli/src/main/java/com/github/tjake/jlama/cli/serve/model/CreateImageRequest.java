package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CreateImageRequestModel;
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



@JsonTypeName("CreateImageRequest")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateImageRequest   {
  private @Valid String prompt;
  private @Valid CreateImageRequestModel model;
  private @Valid Integer n = 1;
  public enum QualityEnum {

    STANDARD(String.valueOf("standard")), HD(String.valueOf("hd"));


    private String value;

    QualityEnum (String v) {
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
    public static QualityEnum fromString(String s) {
        for (QualityEnum b : QualityEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected string value '" + s + "'");
    }

    @JsonCreator
    public static QualityEnum fromValue(String value) {
        for (QualityEnum b : QualityEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }
}

  private @Valid QualityEnum quality = QualityEnum.STANDARD;
  public enum ResponseFormatEnum {

    URL(String.valueOf("url")), B64_JSON(String.valueOf("b64_json"));


    private String value;

    ResponseFormatEnum (String v) {
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
    public static ResponseFormatEnum fromString(String s) {
        for (ResponseFormatEnum b : ResponseFormatEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        return null;
    }

    @JsonCreator
    public static ResponseFormatEnum fromValue(String value) {
        for (ResponseFormatEnum b : ResponseFormatEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        return null;
    }
}

  private @Valid ResponseFormatEnum responseFormat = ResponseFormatEnum.URL;
  public enum SizeEnum {

    _256X256(String.valueOf("256x256")), _512X512(String.valueOf("512x512")), _1024X1024(String.valueOf("1024x1024")), _1792X1024(String.valueOf("1792x1024")), _1024X1792(String.valueOf("1024x1792"));


    private String value;

    SizeEnum (String v) {
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
    public static SizeEnum fromString(String s) {
        for (SizeEnum b : SizeEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        return null;
    }

    @JsonCreator
    public static SizeEnum fromValue(String value) {
        for (SizeEnum b : SizeEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        return null;
    }
}

  private @Valid SizeEnum size = SizeEnum._1024X1024;
  public enum StyleEnum {

    VIVID(String.valueOf("vivid")), NATURAL(String.valueOf("natural"));


    private String value;

    StyleEnum (String v) {
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
    public static StyleEnum fromString(String s) {
        for (StyleEnum b : StyleEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        return null;
    }

    @JsonCreator
    public static StyleEnum fromValue(String value) {
        for (StyleEnum b : StyleEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        return null;
    }
}

  private @Valid StyleEnum style = StyleEnum.VIVID;

  /**
   * A text description of the desired image(s). The maximum length is 1000 characters for &#x60;dall-e-2&#x60; and 4000 characters for &#x60;dall-e-3&#x60;.
   **/
  public CreateImageRequest prompt(String prompt) {
    this.prompt = prompt;
    return this;
  }

  
  @ApiModelProperty(example = "A cute baby sea otter", required = true, value = "A text description of the desired image(s). The maximum length is 1000 characters for `dall-e-2` and 4000 characters for `dall-e-3`.")
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
   **/
  public CreateImageRequest model(CreateImageRequestModel model) {
    this.model = model;
    return this;
  }

  
  @ApiModelProperty(value = "")
  @JsonProperty("model")
  public CreateImageRequestModel getModel() {
    return model;
  }

  @JsonProperty("model")
  public void setModel(CreateImageRequestModel model) {
    this.model = model;
  }

  /**
   * The number of images to generate. Must be between 1 and 10. For &#x60;dall-e-3&#x60;, only &#x60;n&#x3D;1&#x60; is supported.
   * minimum: 1
   * maximum: 10
   **/
  public CreateImageRequest n(Integer n) {
    this.n = n;
    return this;
  }

  
  @ApiModelProperty(example = "1", value = "The number of images to generate. Must be between 1 and 10. For `dall-e-3`, only `n=1` is supported.")
  @JsonProperty("n")
 @Min(1) @Max(10)  public Integer getN() {
    return n;
  }

  @JsonProperty("n")
  public void setN(Integer n) {
    this.n = n;
  }

  /**
   * The quality of the image that will be generated. &#x60;hd&#x60; creates images with finer details and greater consistency across the image. This param is only supported for &#x60;dall-e-3&#x60;.
   **/
  public CreateImageRequest quality(QualityEnum quality) {
    this.quality = quality;
    return this;
  }

  
  @ApiModelProperty(example = "standard", value = "The quality of the image that will be generated. `hd` creates images with finer details and greater consistency across the image. This param is only supported for `dall-e-3`.")
  @JsonProperty("quality")
  public QualityEnum getQuality() {
    return quality;
  }

  @JsonProperty("quality")
  public void setQuality(QualityEnum quality) {
    this.quality = quality;
  }

  /**
   * The format in which the generated images are returned. Must be one of &#x60;url&#x60; or &#x60;b64_json&#x60;.
   **/
  public CreateImageRequest responseFormat(ResponseFormatEnum responseFormat) {
    this.responseFormat = responseFormat;
    return this;
  }

  
  @ApiModelProperty(example = "url", value = "The format in which the generated images are returned. Must be one of `url` or `b64_json`.")
  @JsonProperty("response_format")
  public ResponseFormatEnum getResponseFormat() {
    return responseFormat;
  }

  @JsonProperty("response_format")
  public void setResponseFormat(ResponseFormatEnum responseFormat) {
    this.responseFormat = responseFormat;
  }

  /**
   * The size of the generated images. Must be one of &#x60;256x256&#x60;, &#x60;512x512&#x60;, or &#x60;1024x1024&#x60; for &#x60;dall-e-2&#x60;. Must be one of &#x60;1024x1024&#x60;, &#x60;1792x1024&#x60;, or &#x60;1024x1792&#x60; for &#x60;dall-e-3&#x60; models.
   **/
  public CreateImageRequest size(SizeEnum size) {
    this.size = size;
    return this;
  }

  
  @ApiModelProperty(example = "1024x1024", value = "The size of the generated images. Must be one of `256x256`, `512x512`, or `1024x1024` for `dall-e-2`. Must be one of `1024x1024`, `1792x1024`, or `1024x1792` for `dall-e-3` models.")
  @JsonProperty("size")
  public SizeEnum getSize() {
    return size;
  }

  @JsonProperty("size")
  public void setSize(SizeEnum size) {
    this.size = size;
  }

  /**
   * The style of the generated images. Must be one of &#x60;vivid&#x60; or &#x60;natural&#x60;. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images. This param is only supported for &#x60;dall-e-3&#x60;.
   **/
  public CreateImageRequest style(StyleEnum style) {
    this.style = style;
    return this;
  }

  
  @ApiModelProperty(example = "vivid", value = "The style of the generated images. Must be one of `vivid` or `natural`. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images. This param is only supported for `dall-e-3`.")
  @JsonProperty("style")
  public StyleEnum getStyle() {
    return style;
  }

  @JsonProperty("style")
  public void setStyle(StyleEnum style) {
    this.style = style;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateImageRequest createImageRequest = (CreateImageRequest) o;
    return Objects.equals(this.prompt, createImageRequest.prompt) &&
        Objects.equals(this.model, createImageRequest.model) &&
        Objects.equals(this.n, createImageRequest.n) &&
        Objects.equals(this.quality, createImageRequest.quality) &&
        Objects.equals(this.responseFormat, createImageRequest.responseFormat) &&
        Objects.equals(this.size, createImageRequest.size) &&
        Objects.equals(this.style, createImageRequest.style);
  }

  @Override
  public int hashCode() {
    return Objects.hash(prompt, model, n, quality, responseFormat, size, style);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateImageRequest {\n");
    
    sb.append("    prompt: ").append(toIndentedString(prompt)).append("\n");
    sb.append("    model: ").append(toIndentedString(model)).append("\n");
    sb.append("    n: ").append(toIndentedString(n)).append("\n");
    sb.append("    quality: ").append(toIndentedString(quality)).append("\n");
    sb.append("    responseFormat: ").append(toIndentedString(responseFormat)).append("\n");
    sb.append("    size: ").append(toIndentedString(size)).append("\n");
    sb.append("    style: ").append(toIndentedString(style)).append("\n");
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

