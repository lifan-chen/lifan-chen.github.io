---
title: ChatGLM
categories:
  - Note
  - AI
  - NLP
abbrlink: 62616
date: 2023-10-17 16:39:22
---

## 1 Title

GLM: General Language Model Pretraining with Autoregressive Blank Infilling

GLM-130B: AN OPEN BILINGUAL PRE-TRAINED MODEL

## 2 BK

* Various types of pretraining architectures
    * autoencoding models: BERT
        * learn bidrectional context encoders via denoising objectives
        * the encoders produce contextualized representation that suit natural language understanding tasks, but could not be directly applied for text generation.
    * autoregressive models: GPT
        * learn left-to-right language models
        * succeed in long-text generation and show few-shot learning ability when scaled to billions of parameters
        * the inherent disadvantage is the unidirectional attention mechanism, which cannot fully capture the dependencies between the context words in NLU tasks (不能完全捕捉NLU任务中上下文单词之间的依赖关系).
    * encoder-decoder models: T5
        * adopt bidirectional attention for the encoder, unidirectional attention for the decoder, and cross attention between them.
        * typically deployed in conditional generation tasks, such as text summarization and response generation.
* None of the pretraining frameworks performs the best for all tasks of three main categories:
    * natural language understanding (NLU), unconditional generation, conditional generation
    * previous works have tried to unify different frameworks by combining their objectives via multi-task learning (since the autoencoding and autoregressive objectives differ by nature, a simple unification cannot fully inherit the advantages of both frameworks)
* Typically, for downstream NLU tasks, a linear classifier takes the representations of sequences or tokens produced by pretrained models as input and predicts the correct labels. The practices are different from the generative pretraining and finetuning.

## 3 Intro

* We propose a General Language Model (GLM)

    * based on autoregressive blank infilling
        * randomly blank out continuous spans of tokens from the input text (随机地从输入文本中删除连续跨度的token), following the idea of autoencoding
        * train the model to sequentially reconstruct the spans (训练模型依次重构跨度), following the idea of autoregressive pretraining

    * improves blank filling pretraining by add 2D postional encodings and allowing an arbitrary order to predict spans

* We reformulate NLU tasks as manually-crafted cloze questions that mimic human language.

* We show that by varying the number and lengths of missing spans, the autoregressive blank filling objective can pretrain language models for conditional and unconditional generation.

## 4 Methods

GLM formulates NLU tasks as cloze questions that contain task descriptions, which can be answered by autoregressive generation.

### 4.1 Autoregressive Blank Infilling

GLM is training by optimizing an *autoregressive blank infilling* objective.

Given an input text $x = [x_1, \cdots, x_n]$, multiple text spans ${s_1, \cdots, s_m}$ are sampled, where each $s_i$ corresponds to a series of consecutive tokens $[s_{i, 1}, \cdots, s_{i, l_i}]$ in $x$. Each span is replaced with a single [MASK] token, forming a corrupted text $x_{corrupt}$.

The model predicts the missing tokens in the spans from the corrupted(毁坏的) text in an autoregressive manner, which means when predicting the missing tokens in a span, the model has access to the corrupted text and the previously predicted spans.

To fully capture the interdependencies between different spans, we randomly permute the order of the spans, similar to the permutation language model. 为了充分捕获不同跨度之间的相互依赖关系，我们对跨度的顺序进行随机置换，类似于置乱语言模型。Formally, let $Z_m$ be the set of all possible permutations of the length-m index sequence $[1, 2, \cdots, m]$, and $s_{i_{<i}}$ be $[s_{z_1}, \cdots, s_{z_{i-1}}]$, we define the pretraining objective as 
$$
\mathop{max}\limits_{\theta}\mathbb{E}_{z \sim Zm}
\left[
	\sum\limits_{i=1}^n log\;p_{\theta}(s_{z_i}|x_{corrupt, s_{z_{<i}}})
\right]
$$

We always generate the tokens in each blank following a left-to-right order, i.e. the probability of generating the span $s_i$ is factorized as :
$$
p_{\theta}(s_i|x_{corrupt}, s_{z_{<i}})\\
= \prod \limits_{j=1}^{l_i}p(s_{i,j}|x_{corrupt}, s_{z_{<i}}, s_{i,<j})
$$

* We implement the autoregressive blank infilling objective with the following techniques
    * the input $x$ is divided into two parts
        * Part A is the corrupted text $x_{corrupt}$ and Part B consists of the masked spans.
        * Part A tokens can attend to each other, but cannot attend to any tokens in B.
        * Part B tokens can attend to Part A and antecedents in B, but cannot attend to any subsequent token in B.
    * To enable autoregressive generation, each span is padded with special tokens [START] and [END], for input and output respectively.
    * In this way, our model automatically learns a bidirectional encoder (for Part A) and a unidirectional decoder (for Part B) in a unified model.
    * We randomly sample spans of length drawn from a Poisson distribution with $\lambda = 3$. We repeatedly sample new spans until at least 15% of the original tokens are masked.

### 4.2 Multi-Task Pretraining

We study a *multi-task pretraining* setup, in which a second objective of generating longer text is jointly optimized with the blank infilling objective.

* We consider the following two objectives

    * Document-level.

        We sample a single span whose length is sampled from a uniform distribution over 50%-100% of the original length. 

        The objective aims for long text generation.

    * Sentence-level.

        We restrict that the masked spans must be full sentences. Multiple spans (sentences) are sampled to cover 15% of the original tokens. 

        This objective aims for seq2seq tasks whose predictions are often complete sentences or paragraphs.

Both new objectives are defined in the same way as the original objective, i.e. Eq. 1. The only difference is the number of spans and the span lengths.

## 5 Model Architecture

* GLM uses a single Transformer with several modifications to the architecture.
    * we rearrange the order of layer normalization and the residual connection (重新排列了层归一化和残差连接的顺序), which has been shown critical for large-scale language models to avoid numerical errors.
    * we use a single linear layer for the output token prediction
    * we replace ReLU activation functions with GeLUs

### 5.1 2D Positional Encoding

One of the challenges of the autoregressive blank infilling task is how to encode the positional information. Transformers rely on positional encodings to inject the absolute and relative positions of the token.

* We propose 2D positional encodings. 

    * Specifically, each token is encoded with two positional ids.

    * The first positional id represents the position in the corrupted text $x_{corrupt}$.

        For the masked spans, it is the position of the corresponding [MASK] token. 对于被掩蔽的span，它是对应的[ MASK ] token的位置。

    * The second positional id represents the intra-span position.

        * For tokens in Part A, their second positional ids are 0.
        * For tokens in Part B, they range from 1 to the length of the span.

    * The two positional ids are projected into two vectors via learnable embedding tables, which are both added to the input token embeddings. 两个位置id通过可学习的嵌入表投影为两个向量，这两个向量都被添加到输入令牌嵌入中.

* Our encoding method ensures that the model is not aware of the length of the masked span when reconstructing them. 我们的编码方法可确保模型在重建时不知道屏蔽跨度的长度。

    Our design fits downstream tasks as usually the length of the generated text is unknown beforehand.

## 6 Conclusion

* GLM is a general pretraining framework for natural language understanding and generation.
* We show that the NLU tasks can be formulated as conditional generation task, and therefore solvable by autoregressive models. 我们证明了NLU任务可以被描述为条件生成任务，因此可以通过自回归模型来解决
* GLM unifies the pretraining objectives for different tasks as autoregressive blank infilling, with mixed attention masks and the novel 2D position encodings.

