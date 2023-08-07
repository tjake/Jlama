# Jlama / A LLM inference engine for Java

## Introduction

Jlama is a pure Java implementation of a LLM inference engine.

It currently supports the following models and formats:

  * Llama & Llama2
  * GPT-2 
  * Huggingface [SafeTensors](https://github.com/huggingface/safetensors) Model format
  * Support for Float16 and Float32 models

This project is a work in progress.  
There are a lot of things that need to be done, but it is functional.

Jlama is built with Java 20 and utilizes the new [Vector API](https://openjdk.java.net/jeps/338) 
for faster inference.

## Why?

Oh you know... just for fun.  This should be helpful for anyone who wants to understand how LLMs work, 
or wants to use them in a Java project.

## How to use
As of now the best way to use this is to look at the [TestModels](...) Unit Tests.

First download the [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
model and put it in the `data` directory.

You can also download one on the gpt2 models like [gpt2-medium](https://huggingface.co/gpt2-medium)

```shell
./mvnw test -Dtest=TestModels#GPT2Run
./mvnw test -Dtest=TestModels#LlamaRun
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
and energy. Both theories were developed by Albert Einstein in the early 20th century and have been widely accepted 
and experimentally confirmed since then. The theory of relativity has had a profound impact on our understanding of the
universe, from the smallest subatomic particles to the largest structures of the cosmos.
Here are some key points to understand about the theory of relativity:

Time dilation: The theory of special relativity shows that time appears to pass more slowly for an observer in motion 
relative to a stationary observer. This is known as time dilation.

Length contraction: The theory of special relativity also shows that objects appear shorter to an observer in 
motion relative to a stationary observer. This is known as length contraction.

Relativity of simultaneity: The theory of 

elapsed: 204s, 798.289063ms per token
```