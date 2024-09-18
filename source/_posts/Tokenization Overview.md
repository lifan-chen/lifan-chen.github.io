---
title: Tokenization Operation
categories:
  - Note
  - AI
  - NLP
  - tokenizer
abbrlink: 5045
date: 2023-10-20 16:07:50
---

in NLP, most of the data is raw text

the tokenizer is used to transform raw text to numbers (a meaningful representation)

## Subword-based tokenization

* the flaws of word-based and character-based tokenization
    * Word-based tokenization
        * very large vocabularies
        * large quantity of out-of-vocabulary tokens
        * loss of meaning across very similar words
    * Character-based tokenization
        * very long sequences
        * less meaningful individual tokens

Subword-tokenization lies in between character-based and word-based tokenization algorithms. 

The idea is to find a middle ground between very large vocabularies, large quantity of out-of-vocabulary tokens, loss of meaning across very similar words, for word-based tokenizers and very long sequences, less meaningful individual tokens for character-based tokenizers. 这个想法是在非常大的词汇表、大量词汇外标记、非常相似的单词之间失去意义、基于单词的标记生成器和非常长的序列、基于字符的标记生成器的有意义的单个标记之间找到一个中间立场 。

* Frequently used words should not be split into smaller subwords.
* Rare words should be decomposed into meaningful subwords.

<img src="./image-20231020163748835.png" alt="image-20231020163748835" style="zoom:50%;" />

Subwords help identify similar syntactic or semantic situations in text.

<img src="./image-20231020163919035.png" alt="image-20231020163919035" style="zoom:50%;" />

Subword tokenization algorithms can identify start of word tokens.

Most models obtaining state-of-the-art results in English today use some kind of subword-tokenization algorithm

* WordPiece: BERT, DistilBERT
* Unigram: XLNet, ALBERT
* Byte-Pair Encoding: GPT-2, RoBERTa

## Word-based tokenization

* the text is split on spaces

    The model has representations that are based on entire words.

* each word has a specific ID

    The information held in a single number is high as a word contains a lot of contextual and semantic information in a sentence. 单个数字所包含的信息量很大，因为一个单词在句子中包含大量上下文和语义信息。

* very similar words have entirely different meanings

* the vocabulary can end up very large (large vocabularies result in heavy models)

    We can limit the amount of words we add to the vocabulary. Out of vocabulary words result in a loss information

## Character-based tokenization

* vocabularies are similar

* fewer out of vocabulary words

    As our vocabulary contains all characters used in a language, even words unseen during the tokenizer training can still can be tokenized, so out-of-vocabulary tokens will be less frequent.

* characters do not hold as much information individually as a word would hold

## The Tokenization Pipeline

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Let's try to tokenize!")
print(inputs["input_ids"])

>>> [101, 2292, 1005, 1055, 3046, 2000, 19204, 4697, 999, 102]
```

* The tokenization pipeline: from input text to a list of numbers

    Raw text -> Tokens -> Special tokens -> Input IDs

    * Tokens: words, parts of words, or punctuation symbols

        lowercasing all words, follow a set of rules to split the result in small chunks of text (Most of the Transformers models use a subword tokenization algorithm, which means that one given word can be split in several tokens)

        The ## prefix in front of "ize" is the convention used by BERT to indicate this token is not the beginning of a word. (other tokenizers may use different conventions)

        ```python
        tokens = tokenizer.tokenize("Let's try to tokenize!")
        print(tokens)
        >>> ['let', "'", 's', 'try', 'to', 'token', '##ize', '!']
        ```

    * map those tokens to their respective IDs as defined by the vocabulary of the tokenizer.

        ```python
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(input_ids)
        >>> [2292, 1005, 1055, 3046, 2000, 19204, 4697, 999]
        ```

    * 对比最上面的输出，头尾都分别有一些数字缺失了，这些缺失的数字是 the special tokens

        The special tokens are added by the prepare_for_model method, which knows the indices of those tokens in the vocabulary and just adds the proper numbers. 特殊标记是通过prepare_for_model方法添加的，该方法知道词汇表中这些标记的索引，并且只添加适当的数字。

        ```python
        final_ids = tokenizer.prepare_for_model(input_ids)
        print(final_ids['input_ids'])
        >>> [101, 2292, 1005, 1055, 3046, 2000, 19204, 4697, 999, 102]
        ```

        look at special tokens by using the decode method on the outputs of the tokenizer object

        As for the prefix for beginning of words/part of words, those special tokens vary depending on which tokenizer you are using.

        ```python
        print(tokenizer.decode(final_ids['input_ids']))
        >>> "[CLS] let's try to tokenize! [SEP]"
        ```

        

