package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.CompletionUsage;
import com.github.tjake.jlama.cli.serve.model.CreateEditResponseChoicesInner;
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



@JsonTypeName("CreateEditResponse")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class CreateEditResponse   {
  private @Valid List<@Valid CreateEditResponseChoicesInner> choices = new ArrayList<>();
  public enum ObjectEnum {

    EDIT(String.valueOf("edit"));


    private String value;

    ObjectEnum (String v) {
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
    public static ObjectEnum fromString(String s) {
        for (ObjectEnum b : ObjectEnum.values()) {
            // using Objects.toString() to be safe if value type non-object type
            // because types like 'int' etc. will be auto-boxed
            if (java.util.Objects.toString(b.value).equals(s)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected string value '" + s + "'");
    }

    @JsonCreator
    public static ObjectEnum fromValue(String value) {
        for (ObjectEnum b : ObjectEnum.values()) {
            if (b.value.equals(value)) {
                return b;
            }
        }
        throw new IllegalArgumentException("Unexpected value '" + value + "'");
    }
}

  private @Valid ObjectEnum _object;
  private @Valid Integer created;
  private @Valid CompletionUsage usage;

  /**
   * A list of edit choices. Can be more than one if &#x60;n&#x60; is greater than 1.
   **/
  public CreateEditResponse choices(List<@Valid CreateEditResponseChoicesInner> choices) {
    this.choices = choices;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "A list of edit choices. Can be more than one if `n` is greater than 1.")
  @JsonProperty("choices")
  @NotNull
  public List<CreateEditResponseChoicesInner> getChoices() {
    return choices;
  }

  @JsonProperty("choices")
  public void setChoices(List<@Valid CreateEditResponseChoicesInner> choices) {
    this.choices = choices;
  }

  public CreateEditResponse addChoicesItem(CreateEditResponseChoicesInner choicesItem) {
    if (this.choices == null) {
      this.choices = new ArrayList<>();
    }

    this.choices.add(choicesItem);
    return this;
  }

  public CreateEditResponse removeChoicesItem(CreateEditResponseChoicesInner choicesItem) {
    if (choicesItem != null && this.choices != null) {
      this.choices.remove(choicesItem);
    }

    return this;
  }
  /**
   * The object type, which is always &#x60;edit&#x60;.
   **/
  public CreateEditResponse _object(ObjectEnum _object) {
    this._object = _object;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The object type, which is always `edit`.")
  @JsonProperty("object")
  @NotNull
  public ObjectEnum getObject() {
    return _object;
  }

  @JsonProperty("object")
  public void setObject(ObjectEnum _object) {
    this._object = _object;
  }

  /**
   * The Unix timestamp (in seconds) of when the edit was created.
   **/
  public CreateEditResponse created(Integer created) {
    this.created = created;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "The Unix timestamp (in seconds) of when the edit was created.")
  @JsonProperty("created")
  @NotNull
  public Integer getCreated() {
    return created;
  }

  @JsonProperty("created")
  public void setCreated(Integer created) {
    this.created = created;
  }

  /**
   **/
  public CreateEditResponse usage(CompletionUsage usage) {
    this.usage = usage;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("usage")
  @NotNull
  public CompletionUsage getUsage() {
    return usage;
  }

  @JsonProperty("usage")
  public void setUsage(CompletionUsage usage) {
    this.usage = usage;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    CreateEditResponse createEditResponse = (CreateEditResponse) o;
    return Objects.equals(this.choices, createEditResponse.choices) &&
        Objects.equals(this._object, createEditResponse._object) &&
        Objects.equals(this.created, createEditResponse.created) &&
        Objects.equals(this.usage, createEditResponse.usage);
  }

  @Override
  public int hashCode() {
    return Objects.hash(choices, _object, created, usage);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class CreateEditResponse {\n");
    
    sb.append("    choices: ").append(toIndentedString(choices)).append("\n");
    sb.append("    _object: ").append(toIndentedString(_object)).append("\n");
    sb.append("    created: ").append(toIndentedString(created)).append("\n");
    sb.append("    usage: ").append(toIndentedString(usage)).append("\n");
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

