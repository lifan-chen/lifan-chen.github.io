---
title: P-Tuning
categories:
  - Note
  - AI
  - NLP
abbrlink: 24537
date: 2023-10-24 16:18:52
---

## Prefix-Tuning

Paper: Prefix-Tuning: Optimizing Continuous Prompts for Generation

Fine-tuning is the de facto way to leverage large pretrained language models to perform downstream tasks. 

* It modifies all the language model parameters
* necessitates storing a full copy for each task.
    * This can be prohibitively expensive, given the large size of current LMs.

Propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen, but optimizes a small *continuous task-specific* vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if were "virtual tokens". By learning only 0.1% of the parameters, prefix-tuning obtains comparable performance in the full data setting, outperforms fine-tuning in low-data settings, and extrapolates better to examples with topic unseen during training.

* lightweight fine-tuning: freezes most of the pretrained parameters and augments the model with small trainable modules

    * adapter-tuning: inserts additional task-specific layers between the layers of pretrained language models

* GPT-3 can be deployed without any task-specific tuning. Instead, users prepend a natural language task in struction and a few examples to task input; then generate the output from the LM. (This approach is know as in-context learning or *prompting*)


## P-Tuning v2

Prompt tuning, which only tunes continuous prompts with a frozen language model, substantially reduces per-task storage and memory usage at training.

* However
    * In the context of NLU, prior work reveals that prompt tuning does not perform well for normal-sized pretrained models.
    * Existing methods of prompt tuning cannot handle hard sequence labeling tasks, indicating a lack of universality.

We present a novel empirical finding that properly optimized prompt tuning can be universally effective across a wide range of model scales and NLU tasks. It matches the performance of finetuning while having only 0.1%-3% tuned parameters. (Our method P-Tuning v2 is an implementation of Deep Prompt Tuning optimized and adapted for NLU. Given the universality and simplicity of P-Tuning v2, we believe it can serve as an alternative to finetuning and a strong baseline for future research. ==P-Tuning和P-Tuning v2到底啥关系？？？==)

* Fine-tuning, a widely-used method, updates the entire set of model parameters for a target task.
    * While fine-tuning obtains good performance, it is memory-consuming during training because gradients and optimizer states for all parameters must be stored. Moreover, keeping a copy of model parameters for each task during inference is inconvenient since pre-trained models are usually large.

* ==Prompting== freezes all parameters of a pre-trained model and uses a natural language prompt to query a language model. Prompting requires no training at all and stores one single copy of model parameters.
    * However, **discrete prompting** can lead to suboptimal performance in many cases compared to fine-tuning.

* ==Prompt tuning== is an idea of tuning only the **continuous prompts**. (Prompt tuning 和 Prompting是不是不一样？一个离散的，一个连续的？) 
    * Liu et al.; Lester et al. proposed to add trainable continuous embeddings (also called continuous prompts) to the original sequence of input word embeddings.
    * Only the continuous prompts are updated during training.
    * While prompt tuning improves over prompting on many tasks, it still underperforms fine-tuning while the model size is not large, specifically less than 10 billion parameters. Moreover, as shown in our experiments, prompt tuning performs poorly compared to fine-tuning on several hard sequence labeling tasks such as extractive question answering.
    * Let $\mathcal{V}$ be the vocabulary of a language model $\mathcal{M}$ and let $\mathrm{e}$ be the embedding layer of $\mathcal{M}$.
        *  In the case of discrete prompting, prompt tokens  {"It", "is", "[MASK]"}$\;\in\;\mathcal{V}$ can be used to classify a movie review.
        * Given the trainable continuous embeddings $[h_0, \dots, h_i]$, in the input embedding sequence is written as $[\mathrm{e(x)}, h_0,\dots ,h_i, \mathrm{e}(''[MASK]'')]$.
    * Lack of universality across scales
        * prompt tuning can be comparable to fine-tuning when the model scales to over 10 billion parameters.
        * However, for medium-sized models (from 100M to 1B) that are widely used, prompt tuning performs much worse than fine-tuning.
    * Lack of universality across tasks
        * superiority on some of the NLU benchmarks
        * perform poorly on typical sequence tagging tasks compared to fine-tuning.

Our main contribution in this paper is a novel empirical finding that properly optimized prompt tuning can be comparable to fine-tuning universally across various model scales and NLU tasks. 我们在本文中的主要贡献是一个新的经验发现，适当优化的即时微调可以与不同模型规模和NLU任务中的普遍微调相媲美.

Our approach P-tuning v2 is not conceptually novel. It can be viewed as an optimized and adapted implementation of **Deep Prompt Tuning** (Prefix-tuning: Optimizing continuous prompts for generation; Learning how to ask: Querying lms with mixtures of soft prompts) designed for generation and knowledge probing (为生成和知识探测而设计的). 

* Deep prompt tuning increases the capacity of continuous prompts and closes the gap to fine-tuning across various settings, especially for small models and hard tasks.

The most significant improvement originates from appling continuous prompts for every layer of the pretrained model, instead of the mere input layer. 最显著的改进来自于对预训练模型的每一层施加连续的提示，而不是仅仅输入层

Considering these challenges, we propose P-tuning v2, which adapts deep prompt tuning as a universal solution across scales and NLU tasks.

* Deep Prompt Tuning
    * continuous prompts are only inserted into the input embedding sequence
    * two challenges: First, the number of tunable parameters is limited due to the constraints of sequence length. Second, the input embeddings have relatively indirect impact on model predictions. 首先，由于序列长度的限制，可调参数的数量有限。第二，输入嵌入对模型预测具有相对间接的影响。

To address these challenges, P-tuning v2 employs the idea of deep prompt tuning. Prompts in different layers are added as prefix tokens.

One one hand, P-tuning v2 have more tunable task-specific parameters (from 0.01% to 0.1% - 3%) to allow more per-task capacity while being parameter-efficient; on the other hand, prompts added to deeper layers have more direct impact on predictions.

* P-tuning v2: Across Scales
    * P-tuning v2 matches the fine-tuning performance in all the tasks at a smaller scale.
    * even significantly outperforms fine-tuning on RTE
    * P-tuning v2 is always comparable to fine-tuning at all scales but with only 0.1% task-specific parameters needed comparing to fine-tuning.
* P-tuning v2: Across Tasks
    * P-tuning v2 can be generally comparable to fine-tuning on all tasks.

### Optimization and Implementation

* Reparameterization 再参数化
    * Prior works usually leverage a reparameterization encoder such as MLP to transform trainable embeddings. 以往的工作通常利用一个重新参数化的编码器，如MLP，来转换可训练的嵌入。
    * For NLU, we discover that its usefulness depends on tasks and datasets.
* Prompt Length
    * different NLU tasks usually achieve their best performance with different prompt lengths
* Multi-task Learning
    * Multi-task is optional for P-Tuning v2 but can be used for further boost performance by providing a better initialization.
* Classification Head ==这啥意思？==
    * Using a language modeling head to predict verbalizers has been central for prompt tuning, but we find it unnecessary in a full-data setting and incompatible with sequence labeling.
    * P-tuning v2 instead applies a randomly-initialized classification head on top of the tokens as in BERT.

### Experiments

In this work, all methods except for fine-tuning are conducted with frozen language model backbones.

Our experiments are all conducted in the fully-supervised setting rather than few-shot setting.
