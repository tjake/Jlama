#!/usr/bin/env bash

set -o pipefail

function usage() {
cat << EOF
    usage: $0 [-h] [-a HF_ACCESS_TOKEN] owner/model_name

    This script will download a safetensor files and inference configuration from huggingface.


    OPTIONS:
       -a   Specify a HF access token when downloading restricted models. 
            You may also set the HF_ACCESS_TOKEN environment variable.
            To create a token see https://huggingface.co/settings/tokens
       
       -h   Show this message

    EXAMPLES:
        $0 gpt2-medium
        $0 -a XXXXXXXXX meta-llama/Llama-2-7b-chat-hf 
EOF
}

HF_ACCESS_TOKEN=$HF_ACCESS_TOKEN

while getopts ":ha:" OPTION
do
  case $OPTION in
    h)
      usage
      exit 1
      ;;
    a)
      HF_ACCESS_TOKEN=$OPTARG
      ;;
    ?)
      usage
      exit 1
      ;;
  esac
done

# Shift off the options and optional arguments
shift $((OPTIND-1))

# Check if Model is provided
if [ "$#" -ne 1 ]; then
    usage
    exit 1
fi

MODEL="$1"
HTTP_HEADER=
if [ ! -z $HF_ACCESS_TOKEN ]; then
   HTTP_HEADER="-H 'Authorization: Bearer ${HF_ACCESS_TOKEN}'"
fi

# Check if curl is installed
if ! command -v curl &>/dev/null; then
    echo "Error: curl is not installed. Please install it and try again."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &>/dev/null; then
    echo "Error: jq is not installed. Please install it and try again."
    exit 1
fi

# Fetch information about the model
OUTPUT=$(eval curl -sfS $HTTP_HEADER "https://huggingface.co/api/models/${MODEL}")
EC=$?
if [ $EC -ne 0 ]; then 
    echo "No valid model found or trying to access a restricted model (use -a?)"
    exit 1
fi

FILES=$(echo $OUTPUT | jq -r .siblings[].rfilename)

if [ "${#FILES[@]}" == "0" ]; then
  echo "No valid model found"
  exit 1
fi

ALL_FILES=$(IFS=$'\n' ; echo "${FILES[*]}" | grep safetensor)

if [ "${#ALL_FILES[@]}" == "0" ]; then
  echo "Model is not availible in safetensor format"
  exit 1
fi

MODEL_DIR=$(basename "$MODEL")
mkdir -p $MODEL_DIR

ALL_FILES+=(config.json vocab.json tokenizer.json)

for FILE in ${ALL_FILES[*]}
do
   echo "Downloading ${MODEL_DIR}/${FILE}..."
   eval curl -fL --progress-bar $HTTP_HEADER "https://huggingface.co/${MODEL}/resolve/main/${FILE}" -o "${MODEL_DIR}/${FILE}"
   if [ $? -ne 0 ]; then 
      echo "Encountered error"
   fi
done


echo "Downloading ${MODEL_DIR}/tokenizer.model (if exists)"
eval curl -fL --progress-bar $HTTP_HEADER "https://huggingface.co/${MODEL}/resolve/main/tokenizer.model" -o "${MODEL_DIR}/tokenizer.model"

echo "Done!"
