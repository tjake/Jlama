package com.github.tjake.jlama.cli.serve.model;

import com.github.tjake.jlama.cli.serve.model.MessageFileObject;
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



@JsonTypeName("ListMessageFilesResponse")
@jakarta.annotation.Generated(value = "org.openapitools.codegen.languages.JavaJAXRSSpecServerCodegen", date = "2024-03-05T15:38:54.969215602-05:00[America/New_York]")
public class ListMessageFilesResponse   {
  private @Valid String _object;
  private @Valid List<@Valid MessageFileObject> data = new ArrayList<>();
  private @Valid String firstId;
  private @Valid String lastId;
  private @Valid Boolean hasMore;

  /**
   **/
  public ListMessageFilesResponse _object(String _object) {
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
  public ListMessageFilesResponse data(List<@Valid MessageFileObject> data) {
    this.data = data;
    return this;
  }

  
  @ApiModelProperty(required = true, value = "")
  @JsonProperty("data")
  @NotNull
  public List<MessageFileObject> getData() {
    return data;
  }

  @JsonProperty("data")
  public void setData(List<@Valid MessageFileObject> data) {
    this.data = data;
  }

  public ListMessageFilesResponse addDataItem(MessageFileObject dataItem) {
    if (this.data == null) {
      this.data = new ArrayList<>();
    }

    this.data.add(dataItem);
    return this;
  }

  public ListMessageFilesResponse removeDataItem(MessageFileObject dataItem) {
    if (dataItem != null && this.data != null) {
      this.data.remove(dataItem);
    }

    return this;
  }
  /**
   **/
  public ListMessageFilesResponse firstId(String firstId) {
    this.firstId = firstId;
    return this;
  }

  
  @ApiModelProperty(example = "file-hLBK7PXBv5Lr2NQT7KLY0ag1", required = true, value = "")
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
  public ListMessageFilesResponse lastId(String lastId) {
    this.lastId = lastId;
    return this;
  }

  
  @ApiModelProperty(example = "file-QLoItBbqwyAJEzlTy4y9kOMM", required = true, value = "")
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
  public ListMessageFilesResponse hasMore(Boolean hasMore) {
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
    ListMessageFilesResponse listMessageFilesResponse = (ListMessageFilesResponse) o;
    return Objects.equals(this._object, listMessageFilesResponse._object) &&
        Objects.equals(this.data, listMessageFilesResponse.data) &&
        Objects.equals(this.firstId, listMessageFilesResponse.firstId) &&
        Objects.equals(this.lastId, listMessageFilesResponse.lastId) &&
        Objects.equals(this.hasMore, listMessageFilesResponse.hasMore);
  }

  @Override
  public int hashCode() {
    return Objects.hash(_object, data, firstId, lastId, hasMore);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("class ListMessageFilesResponse {\n");
    
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

