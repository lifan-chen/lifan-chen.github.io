---
title: The Tokenization Pipeline
categories:
  - Note
  - AI
  - NLP
  - tokenizer
abbrlink: 9760
date: 2023-10-20 21:12:30
---

简单使用

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("Let's try to tokenize!")
print(inputs["input_ids"])

>>> [101, 2292, 1005, 1055, 3046, 2000, 19204, 4697, 999, 102]
```

**The tokenization pipeline: from input text to a list of numbers**

Raw text -> Tokens -> Special tokens -> Input IDs

1. Tokens: words, parts of words, or punctuation symbols

lowercasing all words, follow a set of rules to split the result in small chunks of text (Most of the Transformers models use a subword tokenization algorithm, which means that one given word can be split in several tokens)

The ## prefix in front of "ize" is the convention used by BERT to indicate this token is not the beginning of a word. (other tokenizers may use different conventions)

```python
tokens = tokenizer.tokenize("Let's try to tokenize!")
print(tokens)
>>> ['let', "'", 's', 'try', 'to', 'token', '##ize', '!']
```

2. map those tokens to their respective IDs as defined by the vocabulary of the tokenizer.

```python
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)
>>> [2292, 1005, 1055, 3046, 2000, 19204, 4697, 999]
```

3. 对比最上面的输出，头尾都分别有一些数字缺失了，这些缺失的数字是 the special tokens

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



