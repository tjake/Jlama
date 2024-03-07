package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.MessageObject;
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



@JsonTypeName("ListMessagesResponse")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class ListMessagesResponse   {
  private @Valid String _object;
  private @Valid List<@Valid MessageObject> data = new ArrayList<>();
  private @Valid String firstId;
  private @Valid String lastId;
  private @Valid Boolean hasMore;

  /**
   **/
  public ListMessagesResponse _object(String _object) {
    this._object = _object;
    return this;
  }

  
  @ApiModelProperty(example = "list", required = true, value = "")
  @JsonProperty("object")
  @NotNull
  public String getObject() {
    return _object;
  }

  @JsonProperty("object")
  public void setObject(String _object) {
    this._object = _object;
  }

  /**
   **/
  public ListMessagesResponse data(List<@Valid MessageObject> data) {
    this.data = data;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("data")
  @NotNull
  public List<MessageObject> getData() {
    return data;
  }

  @JsonProperty("data")
  public void setData(List<@Valid MessageObject> data) {
    this.data = data;
  }

  public ListMessagesResponse addDataItem(MessageObject dataItem) {
    if (this.data == null) {
      this.data = new ArrayList<>();
    }

    this.data.add(dataItem);
    return this;
  }

  public ListMessagesResponse removeDataItem(MessageObject dataItem) {
    if (dataItem != null && this.data != null) {
      this.data.remove(dataItem);
    }

    return this;
  }
  /**
   **/
  public ListMessagesResponse firstId(String firstId) {
    this.firstId = firstId;
    return this;
  }

  
  @ApiModelProperty(example = "msg_hLBK7PXBv5Lr2NQT7KLY0ag1", required = true, value = "")
  @JsonProperty("first_id")
  @NotNull
  public String getFirstId() {
    return firstId;
  }

  @JsonProperty("first_id")
  public void setFirstId(String firstId) {
    this.firstId = firstId;
  }

  /**
   **/
  public ListMessagesResponse lastId(String lastId) {
    this.lastId = lastId;
    return this;
  }

  
  @ApiModelProperty(example = "msg_QLoItBbqwyAJEzlTy4y9kOMM", required = true, value = "")
  @JsonProperty("last_id")
  @NotNull
  public String getLastId() {
    return lastId;
  }

  @JsonProperty("last_id")
  public void setLastId(String lastId) {
    this.lastId = lastId;
  }

  /**
   **/
  public ListMessagesResponse hasMore(Boolean hasMore) {
    this.hasMore = hasMore;
    return this;
  }

  
  @ApiModelProperty(example = "false", required = true, value = "")
  @JsonProperty("has_more")
  @NotNull
  public Boolean getHasMore() {
    return hasMore;
  }

  @JsonProperty("has_more")
  public void setHasMore(Boolean hasMore) {
    this.hasMore = hasMore;
  }


  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    ListMessagesResponse listMessagesResponse = (ListMessagesResponse) o;
    return Objects.equals(this._object, listMessagesResponse._object) &&
        Objects.equals(this.data, listMessagesResponse.data) &&
        Objects.equals(this.firstId, listMessagesResponse.firstId) &&
        Objects.equals(this.lastId, listMessagesResponse.lastId) &&
        Objects.equals(this.hasMore, listMessagesResponse.hasMore);
  }

  @Override
  public int hashCode() {
    return Objects.hash(_object, data, firstId, lastId, hasMore);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class ListMessagesResponse {\n");
    
    sb.append("    _object: ").append(toIndentedString(_object)).append("\n");
    sb.append("    data: ").append(toIndentedString(data)).append("\n");
    sb.append("    firstId: ").append(toIndentedString(firstId)).append("\n");
    sb.append("    lastId: ").append(toIndentedString(lastId)).append("\n");
    sb.append("    hasMore: ").append(toIndentedString(hasMore)).append("\n");
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

