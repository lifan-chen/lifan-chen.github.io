---
title: Streaming LLM
categories:
  - Note
  - AI
  - NLP
  - LLM
abbrlink: 30047
date: 2024-05-07 18:25:54
---

Paper: https://arxiv.org/abs/2309.17453

Cite as: Xiao G, Tian Y, Chen B, et al. Efficient Streaming Language Models with Attention Sinks[M]. arXiv, 2024.

Code: https://github.com/mit-han-lab/streaming-llm

## Why

Deploying Large Language Models in streaming applications such as multi-round dialogue, where long interactions are expected, is urgently needed but poses two major challenges:

1. During the decoding stage, caching previous token's Key and Value states (KV) consumes extensive memory.
2. Popular LLM cannot generalize to longer texts than the training sequence length.

Window attention: Only the most recent KVs are cached, but we show that it fails when the text length surpasses the cache size.

Attention sink: An interesting phenomenon, that keeping the KV of initial tokens will largely recover the performance of window attention.

## What

In this paper, we first demonstrate that the emergence of *attention sink* is due to the strong attention scores towards initial token as a "sink" even if they are not semantically important.

Based on the above analysis, we introduce StreamingLLM, an efficient framework that enables LLMs trained with a *finite length* window to generalize to *infinite sequence length* without any fine-tuning.
