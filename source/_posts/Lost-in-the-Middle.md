---
title: Lost in the Middle
date: 2024-09-22 15:07:39
categories:
  - Note
  - AI
  - NLP
---

If LM can robustly use information within long input contexts, then their performance should be minimally affected by the position of the relevant information in the input context.

* To better understand why language models struggle to robustly access and use information in their input contexts.

    * study the role of model architecture (decoder-only, )

        * encoder-decoder

            Encoder-decoder models are relatively robust to changes in the position of relevant information within their input context, but only when evaluated on sequences within its training-time sequence length.

            When evaluated on sequences longer than those seen during training, we observe a U-shaped performance curve.

    * study query-aware Contextualization

    * instruction and fine-tuning

        Even base language models show a U-shaped performance curve as we vary the position of relevant information in the input context.

* Results

    Our results indicate that prompting language models with longer contexts is a trade-off--providing the language model with more information may help it perform the downstream task, but it also increase the amount of content that the model must reason over, potentially decreasing accuracy. 

## Multi-document question answering

Control

(i) the input context length by changing the number of documents in the input context (akin to retrieving more or less documents in retrieval-augmented generation)

(ii) the position of the relevant information within the input context by changing the order of the documents to place the relevant document at the beginning, middle or end of the context.

* Experimental Setup

    The model input are (i) a question to answer and (ii) $k$ documents where *exactly* one of the documents contains the answer to the question and $k-1$ ‘‘distractor’’ documents do not.

    This task requires the model to access the document that contains the answer within its input context and use it to answer the question.

* Results

    * Model performance is highest when relevant information occurs at the beginning or end of its input context.

        例如GPT-3.5-Turbo在20文档和30文档的最坏情况下，性能低于没有任何输入文档

    * Extended-context models are not necessarily better at using input context.

## How well Can Language Models Retrieve From Input Contexts?

