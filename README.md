# Jlama / A LLM inference engine for Java

<p align="center">
  <img src="docs/jlama.jpg" width="300" height="300" alt="Cute Llama">
</p>

## Introduction

Jlama is a pure Java implementation of a LLM inference engine.

It currently supports the following models and formats:

  * Llama & Llama2 & CodeLlama
  * GPT-2 
  * BERT
  * Huggingface [SafeTensors](https://github.com/huggingface/safetensors) model format
  * Support for Float16, BFloat16 and Float32 models
  * Q8, Q4, Q5 quantizations

Jlama is built with Java 20 and utilizes the new [Vector API](https://openjdk.org/jeps/448) 
for faster inference.

This project is a work in progress.

## Why?

Helpful for anyone who wants to understand how LLMs work, or wants to use LLMs in a Java project.

CPU based inference needs to be pushed to the limit to see if it can be a viable alternative to GPU based inference.

## How to use
As of now the best way to use this is to look at the [TestModels](https://github.com/tjake/Jlama/blob/main/src/test/java/com/github/tjake/jlama/models/TestModels.java) Unit Tests.

Use the `download_hf_models.sh` script in the data directory to download models from huggingface.

```shell
cd data
./download_hf_model.sh gpt2-medium
./download_hf_model.sh -a XXXXXXXX meta-llama/Llama-2-7b-chat-hf
./download_hf_model.sh intfloat/e5-small-v2
```
Then run the tests with:
```shell
cd ..
./mvnw package -DskipTests
./mvnw test -Dtest=TestModels#GPT2Run
./mvnw test -Dtest=TestModels#LlamaRun
./mvnw test -Dtest=TestModels#BertRun
```
## Caveats
  
 * Tokenization (for now) requires JNI wrappers to SentencePiece and Huggingface tokenizers.

# Examples

## GPT-2 (355M parameters)

```
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, 
in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.

The researchers, who were visiting the Andes Mountains in Peru and Chile, discovered that the unicorns, 
known as the yu-yura, were native to the mountain valley, which is somewhere between the Andes and the Chilean Pyrenees.

The researchers believe that the unicorns were introduced to the valley by a group of forest workers who were looking 
for the mythical unicorn.

The researchers believe that the unicorns, known as the yu-yura, were introduced to the valley by a group of forest 
workers who were looking for the mythical unicorn.

In a research published in The American Journal of Physical Anthropology, the researchers discovered that the unicorns
were native to the mountain valley, which is somewhere between the Andes and the Chilean Pyrenees.

elapsed: 10s, 54.234375ms per token

```

## Llama 2 7B

```
Simply put, the theory of relativity states that time and space are relative and can be affected by gravity and motion.
There are two main components to the theory of relativity:

The theory of special relativity, which shows that time and space are relative and can be affected by speed and motion.

The theory of general relativity, which shows that gravity is the curvature of spacetime caused by the presence of mass
and energy. 

Both theories were developed by Albert Einstein in the early 20th century and have been widely accepted 
and experimentally confirmed since then. 

The theory of relativity has had a profound impact on our understanding of the
universe, from the smallest subatomic particles to the largest structures of the cosmos.

Here are some key points to understand about the theory of relativity:

Time dilation: The theory of special relativity shows that time appears to pass more slowly for an observer in motion 
relative to a stationary observer. This is known as time dilation.

Length contraction: The theory of special relativity also shows that objects appear shorter to an observer in 
motion relative to a stationary observer. This is known as length contraction.

Relativity of simultaneity: The theory of 

elapsed: 204s, 798.289063ms per token
```

## Roadmap

    * Support more models
    * Add pure java tokenizers
    * ~~Support Quantization (e.g. k-quantization)~~
    * Add LoRA support
    * GraalVM support
    * Add distributed inference 
