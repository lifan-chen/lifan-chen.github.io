---
title: Long Contexts
categories:
  - Note
  - AI
  - NLP
abbrlink: 16897
date: 2023-11-28 17:25:33
---





Notes on reading the paper "Lost in the Middle: How Language Models Use Long Contexts[^1]"

# 1 BK

* Language models

Language models have become an import and flexible building block in a variety of user-facing language technologies, including conversational interfaces, search and summarization, and collaborative writing. 

These models perform downstream tasks primarily via prompting: all relevant task specification and data to process is formatted as a textual input context, and the model returns a generated text completion.

* Long contexts

    These **input contexts** can contain thousands of tokens, especially when language models are used to process long documents for when language models are augmented with external information.

    Handling these use-cases requires language models to successfully **operate over long sequences**.

    * Existing language models

        Existing language models are generally implemented with Transformers, which require memory and compute that increases quadratically (呈二次增长) in sequence length.

        As a result, Transformer language models were often trained with relatively small context windows (between 512 - 2048 tokens).

    * Recent improvements

        Recent improvements in hardware and algorithm have resulted in language models with **larger context windows** (e.g., 4096, 32K, and even 100K tokens), but ==it remains unclear how these extended-context language models make use of their input contexts when performing downstream tasks==.

# 2 Analysis

We empirically investigate this question via controlled experiments (通过控制实验对这一问题进行了实证研究) with a variety of state-of-the-art open (MPT-30B-Instruct, LongChat-13B(16K)) and closed (OpenAI's GPT-3.5-Turbo and Anthropic's Claude-1.3) language models in settings that require accessing and using information within an input context (在需要访问和使用输入的上下文信息的环境中).

In particular, our experiments make controlled changes to the input context size and the position of the relevant information within the input context and study their effects on language model performance.

# 3 For Further





[^1]: Liu N F, Lin K, Hewitt J, et al. Lost in the middle: How language models use long contexts[J]. arXiv preprint arXiv:2307.03172, 2023.
