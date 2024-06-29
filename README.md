# 🦙 Jlama: A modern LLM inference engine for Java

<p align="center">
  <img src="docs/jlama.jpg" width="300" height="300" alt="Cute Jlama">
</p>

[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.github.tjake/jlama-core/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.github.tjake/jlama-core)

## 🚀 Features

Model Support:
  * Gemma Models
  * Llama & Llama2 & Llama3 Models
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
  * Support for Q8, Q4 model quantization
  * Fast GEMM operations
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
./run-cli.sh download tjake/llama2-7b-chat-hf-jlama-Q4
./run-cli.sh serve models/llama2-7b-chat-hf-jlama-Q4

```
open browser to http://localhost:8080/ui/index.html

<p align="center">
  <img src="docs/demo.png" alt="Demo chat">
</p>

## 👨‍💻 How to use in your Java project

Add the following [maven](https://central.sonatype.com/artifact/com.github.tjake/jlama-core/) dependencies to your project:

```xml

<dependency>
  <groupId>com.github.tjake</groupId>
  <artifactId>jlama-core</artifactId>
  <version>${jlama.version}</version>
</dependency>

<dependency>
  <groupId>com.github.tjake</groupId>
  <artifactId>jlama-native</artifactId>
  <!-- supports linux-x86_64, macos-x86_64/aarch_64, windows-x86_64 
       Use https://github.com/trustin/os-maven-plugin to detect os and arch -->
  <classifier>${os.detected.name}-${os.detected.arch}</classifier>
  <version>${jlama.version}</version>
</dependency>

```

Then you can use the Model classes to run models:

```java
 public void sample() throws IOException {
    String model = "tjake/TinyLlama-1.1B-Chat-v1.0-Jlama-Q4";
    String workingDirectory = "./models";

    String prompt = "What is the best season to plant avocados?";

    // Downloads the model or just returns the local path if it's already downloaded
    File localModelPath = SafeTensorSupport.maybeDownloadModel(workingDirectory, model);

    // Loads the quantized model and specified use of quantized memory
    AbstractModel m = ModelSupport.loadModel(localModelPath, DType.F32, DType.I8);

    // Checks if the model supports chat prompting and adds prompt in the expected format for this model
    if (m.promptSupport().isPresent()) {
        prompt = m.promptSupport().get().newBuilder()
                .addSystemMessage("You are a helpful chatbot who writes short responses.")
                .addUserMessage(prompt)
                .build();
    }

    System.out.println("Prompt: " + prompt + "\n");
    // Generates a response to the prompt and prints it
    // The api allows for streaming or non-streaming responses
    // The response is generated with a temperature of 0.7 and a max token length of 256
    GenerateResponse r = m.generate(UUID.randomUUID(), prompt, 0.7f, 256, false, (s, f) -> System.out.print(s));
    System.out.println(r.toString());
 }
```

## 🕵️‍♀️ How to use as a local client
Jlama includes a cli tool to run models via the `run-cli.sh` command. 
Before you do that first download one or more models from huggingface.

Use the `./run-cli.sh download` command to download models from huggingface.

```shell
./run-cli.sh download gpt2-medium
./run-cli.sh download -t XXXXXXXX meta-llama/Llama-2-7b-chat-hf
./run-cli.sh download intfloat/e5-small-v2
```

Then run the cli tool to chat with the model or complete a prompt.
Quanitzation is supported with the `-q` flag. Or you can use pre-quantized models
located in my [huggingface repo](https://huggingface.co/tjake).

```shell
./run-cli.sh complete -p "The best part of waking up is " -t 0.7 -tc 16 -q Q4 -wq I8 models/Llama-2-7b-chat-hf
./run-cli.sh chat -s "You are a professional comedian" models/llama2-7b-chat-hf-jlama-Q4
```

## 🧪 Examples
### Llama 2 7B

```
You: Tell me a joke about cats. Include emojis.

Jlama:   Sure, here's a joke for you:
Why did the cat join a band? 🎸🐱
Because he wanted to be the purr-fect drummer! 😹🐾
I hope you found that purr-fectly amusing! 😸🐱

elapsed: 11s, prompt 38.0ms per token, gen 146.2ms per token

You: Another one

Jlama:   Of course! Here's another one:
Why did the cat bring a ball of yarn to the party? 🎉🧶
Because he wanted to have a paw-ty! 😹🎉
I hope that one made you smile! 😊🐱

elapsed: 11s, prompt 26.0ms per token, gen 148.4ms per token
```

## 🗺️ Roadmap

* Support more and more models
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
