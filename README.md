# LLM-J / A LLM inference engine for Java

## Introduction

LLM-J is a pure Java implementation of a LLM inference engine.

It currently supports the following models and formats:

  * GPT-2 (small, medium, large, and XL)
  * Llama & Llama2
  * Huggingface [SafeTensors](https://github.com/huggingface/safetensors) Model format
  * Support for Float16 and Float32 models

This project is a work in progress.  
There are a lot of things that need to be done, but it is functional.

LLM-J is built with Java 20 and utilizes the new [Vector API](https://openjdk.java.net/jeps/338) 
for faster inference.

## Why?

Oh you know... just for fun.  This should be helpful for anyone who wants to understand how LLMs work, 
or wants to use them in a Java project.

## How to use

As of now the best way to use this is to look at the [TestModels](...) Unit Tests.
```shell
./mvnw test -Dtest=TestModels#GPT2Run
./mvnw test -Dtest=TestModels#LlamaRun
```
## Caveats

  * Models must be in safetensor format.  Most huggingface models are not in this format. 
    For larger models you will need to re-process them to use a 2G max chunk size (for now).

  
 * Tokenization (for now) requires JNI wrappers to SentencePiece and Huggingface tokenizers.

# Examples

## GPT-2 (355M parameters)


```
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.
The study, published in the journal PLoS One, demonstrates how animals can communicate with each other in both human- and non-human-like languages. It's the first time the animals have been found to communicate in non-human-like languages, according to the scientists.
"The discovery of unicorns and human-like languages highlights the importance of naturalistic research to understand the evolution of human language and its role in human society," said study lead author, Manuela Ruggero, a scientist at the University of Southern California.
"It may help us understand the evolution of language and how it can be used to understand other species, like humans."
The scientists first looked into the community of the forest unicorns. They found that the animals were living in a group of about 100 individuals, with a total population of about 2,000 individuals.
The researchers also found that the animals had facial features that were similar to those of humans.
"It is also interesting that the animals were able...

elapsed: 27s, 109.670586ms per token
```

## Llama 2 7B

There's clearly a bug somewhere ... but it's still fun to see it.

```
Why did the chicken cross the road? get to the other side!



(A jokevincentre:
action:
☉ Thesus?
In a:to get to the other side?
Hey?

universe?
 Unterscheidung?
Wikispecies: Because it was ateach: Because it was a chicken?
 demande?

live?


 Netherland?

 февation?

□
 Hinweis?
 националь?
 n?
 Rahmen?

elapsed: 103s, 936.585571ms per token
```