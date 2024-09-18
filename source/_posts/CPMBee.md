---
title: CPMBee
categories:
  - Note
  - AI
  - NLP
abbrlink: 64658
date: 2023-09-20 20:42:21
---

## 1 Title

CPM: A large-scale generative Chinese Pre-trained language model

## 2 Abs

* Pre-trained Language Models (PLMs) have proven to be beneficial for various downstream NLP tasks.
* GPT-3, the capacity of few-shot (even zero-shot) learning.
* the Chinese Pre-trained Language Model (CPM), the largest Chinese pre-trained model, with 2.6B parameters and 100GB Chinese training data
* The code and parameters are available at https://github.com/TsinghuaAI/CPM.

## 3 Intro

* PLMs
    * ELMo：first introduces biditectional language models (双向语言模型) to learn contextual word vectors via large-scale pre-training.
    * GPT：apply generative pre-training (生成式预训练) to a Transformer-based language model, which improves natural language understanding on a wide range of benchmarks.
    * BERT：is proposed to pre-train deep bidirectional representations on unlabeled texts by jointly conditioning on **both left and right contexts**.
        * enhance BERT by dynamic masking, parameter sharing and modifying pre-training tasks.
    * introduce external knowledge to language representation learning by auxiliary pre-training tasks (辅助预训练任务).
    * GPT-3
        * proven to be effective in various few-shot (even zero-shot) NLP tasks.
        * address Chinese NLP tasks is still challenging, as the training corpus of GPT-3 is primarily English and the parameters are not publicly available.
* Previous works providing prowerful Chinese pre-trained language models are ==limited due to the model size==.
* CPM
    * a Transformer-based autoregressive language model
    * with 2.6 billion parameters and 100GB Chinese training data
    * the largest Chinese pre-trained language model
    * could facilitate downstream Chinese NLP tasks, such as conversation, essay generation, cloze test and language understanding.
    * experiments on various Chinese NLP tasks demonstrate that CPM achieves strong performance on many NLP tasks in the few-shot (even zero-shot) settings.
* Larger models are more proficient at language generation and language understanding.

## 4 Method

### 4.1 Chinese PLM

* Current model is a left-to-right Transformer decoder (similar to the model architecture of GPT)
* Vocabulary Construction
    * previous works on Chinese pre-trained models usually adopt the sub-word vocabulary of BERT-Chinese, which would split the input text to a character-level sequence.
    * **Chinese words usually contain several characters**, and some important semantic meanings of words would **be lost in the character-level sequence**.
    * We construct a new sub-word vocabulary, containing both words and characters.
* Training Strategy
    * adopt **a large batch size**, since the sparseness of word distributions of Chinese is more serious than that of English 汉语的词分布稀疏性比英语更为严重 (our batch size is two times larger than GPT-3)
    * partition the model across GPUs along the width dimension to make the large-scale training available and reduce data transfer among nodes. 将模型延宽度维度进行跨GPU划分，以实现大规模训练，并减少结点间的数据传输


### 4.2 Data processing

* We construct a new sub-word vocabulary based on the word segmented corpus using *unigram language model*
* **set a special token as the splitter** to make the sub-word process reversible
    * BERT-Chinese is irreversible because it will insert extra spaces between Chinese characters and treat the extra spaces as the same as the original spaces in the text.
* Collect different kinds of texts in pre-training, including encyclopedia, news, novels, and Q&A.
* concatenate different documents together by adding "end of document" token after each document to make full use of the input length. 通过在每个文档后添加“文档末尾”标记，将不同的文档串联在一起

### 4.3 Pre-training details

* the hyper-parameter searching on the learning rate and batch size 学习率和批次大小的超参数搜索
    * set the learning rate as $1.5 \times 10^{-4}$ and the batch size as 3072
    * make the model training more stable
* In the first version, adopt the dense attention (密集注意力机制) and the max sequence length is 1024
    * implement sparse attention in the future
* pre-train model for 20000 steps, and the first 5000 steps are for warm-up
* the optimizer is Adam
* using 64 NVIDIA V100, take two weeks

## 5 Exp

## 6 Conclusion

* In this paper
    * explore to train a large-scale generative language model on Chinese corpora and release CPM.
    * experimental results show that CPM excel in several downsteam Chinese NLP tasks.
* In the future
    * add more training data and increasing the model size.
    * optimize the training framework, such as data-transfer scheme between different nodes, to accelerate pre-training process.
    * reduce the model size by model compression.
    * include diverse data to enhance model performance
    * For text data, add a multi-lingual corpus to ==train a large-scale Chinese-centered multi-lingual language model== (maybe finish this work in VISCpm)
    * explore new learning algorithms to **train a joint model**, which can learn form both texts and knowledge graphs for better general intelligence.



