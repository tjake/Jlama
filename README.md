# 🦙 Jlama: A modern Java inference engine for LLMs

<p align="center">
  <img src="docs/jlama.jpg" width="300" height="300" alt="Cute Jlama">
</p>

## 🚀 Features

Model Support:
  * Llama & Llama2 Models
  * Mistral & Mixtral Models
  * GPT-2 Models
  * BERT Models
  * BPE Tokenizers
  * WordPiece Tokenizers


Implements:
  * Flash Attention
  * Mixture of Experts
  * Huggingface [SafeTensors](https://github.com/huggingface/safetensors) model and tokenizer format
  * Support for F32, F16, BF16 models
  * Support for Q8, Q4, Q5 model quantization
  * Distributed Inference!

Jlama is built with Java 21 and utilizes the new [Vector API](https://openjdk.org/jeps/448) 
for faster inference.

## ⭐ Give us a star!

Like what you see? Please consider giving this a star (★)!

## 🤔 What is it used for? 

Add LLM Inference directly to your Java application.

## 🔬 Demo

Jlama includes a simple UI if you just want to chat with an llm.

```
./download-hf-model.sh tjake/llama2-7b-chat-hf-jlama-Q4
./run-cli.sh serve models/llama2-7b-chat-hf-jlama-Q4

```
open browser to http://localhost:8080/ui/index.html

<p align="center">
  <img src="docs/demo.png" alt="Demo chat">
</p>

## 🕵️‍♀️ How to use
Jlama includes a cli tool to run models via the `run-cli.sh` command. 
Before you do that first download one or more models from huggingface.

Use the `download-hf-models.sh` script in the data directory to download models from huggingface.

```shell
./download-hf-model.sh gpt2-medium
./download-hf-model.sh -a XXXXXXXX meta-llama/Llama-2-7b-chat-hf
./download-hf-model.sh intfloat/e5-small-v2
```

Then run the cli:
```shell
./run-cli.sh complete -p "The best part of waking up is " -t 0.7 -tc 16 -q Q4 -wq I8 models/Llama-2-7b-chat-hf
./run-cli.sh chat -p "Tell me a joke about cats." -t 0.7 -tc 16 -q Q4 -wq I8 models/Llama-2-7b-chat-hf
```
## 🧪 Examples

### Llama 2 7B

```
Here is a poem about cats, incluing emojis: 
This poem uses emojis to add an extra layer of meaning and fun to the text.
Cat, cat, so soft and sweet,
Purring, cuddling, can't be beat. 🐈💕
Fur so soft, eyes so bright,
Playful, curious, such a delight. 😺🔍
Laps so warm, naps so long,
Sleepy, happy, never wrong. 😴😍
Pouncing, chasing, always fun,
Kitty's joy, never done. 🐾🎉
Whiskers twitch, ears so bright,
Cat's magic, pure delight. 🔮💫
With a mew and a purr,
Cat's love, forever sure. 💕🐈
So here's to cats, so dear,
Purrfect, adorable, always near. 💕🐈

elapsed: 37s, 159.518982ms per token
```

### GPT-2 (355M parameters)

```
In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, 
in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.
a long and diverse and interesting story is told in this book. The author writes:
...
the stories of the unicornes seem to be based on the most vivid and vivid imagination; they are the stories of animals that are a kind of 'spirit animal' , a partly-human spiritual animal that speaks in perfect English , and that often keep their language under mysterious and inaccessible circumstances.
...
While the unicorn stories are mostly about animals, they tell us about animals from other animal species. The unicorn stories are remarkable because they tell us about animals that are not animals at all . They speak and sing in perfect English , and they are very much human beings.
...
This book is not about the unicorn. It is not about anything in particular . It is about a brief and distinct group of animal beings who have been called into existence in a particular remote and unexplored valley in the Andes Mountains. They speak perfect English , and they are very human beings.
...
The most surprising thing about the tales of the unicorn

elapsed: 10s, 49.437500ms per token
```

## 🗺️ Roadmap

* Support more models
* <s>Add pure java tokenizers</s>
* <s>Support Quantization (e.g. k-quantization)</s>
* Add LoRA support
* GraalVM support
* <s>Add distributed inference</s>

## 🏷️ License and Citation

The code is available under [Apache License](./LICENSE).

If you find this project helpful in your research, please cite this work at

```
@misc{jlama2024,
    title = {Jlama: A modern Java inference engine for large language models},
    url = {https://github.com/tjake/jlama},
    author = {T Jake Luciani},
    month = {January},
    year = {2024}
}
```
