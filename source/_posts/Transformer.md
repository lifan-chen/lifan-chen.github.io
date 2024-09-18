---
title: Transformer
categories:
  - Note
  - AI
  - NLP
abbrlink: 15915
date: 2023-09-22 21:01:33
---

## 1 相关内容

Attention Is All You Need

The code we used to train and evaluate our models is available at [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor).

博客中用于解释结构的代码，来自于[annotated-transformer](https://github.com/harvardnlp/annotated-transformer/tree/debc9fd747bb2123160a98046ad1c2d4da44a567)

部分讲解来自https://youtu.be/H39Z_720T5s?si=0HuRP3OUtu6FEZAN

## Encoder and Decoder

* Encoder and Decoder can be used together, but they can also be used independently.

* How to work

    * The encoder accepts inputs that represent text. It converts this text, these words, into numerical representations. These numerical representations can also be called **embeddings**, or features.

        The encoder accepts inputs and computes a high-level representation of those inputs. These output are then passed to the decoder.

        * Self-attention mechanisam
        * Bi-directional property

    * The decoder is similar to the encoder. It can also accept the same inputs (text representations) as the encoder.

        * Masked self-attention mechanism
        * Uni-directional property
        * Auto-regressive manner

        The decoder uses the encoder's output alongside other inputs, in order to generate a prediction. It then predicts an output, which it will re-use in future iterations, hence the term "auto-regressive" (然后，它预测一个输出，并将在未来的迭代中重复使用该输出，因此称为“自回归”).

        Additionally to the encoder outputs, we also give the decoder a sequence. When prompting the decoder for an output with no initial sequence, we can give it the value that indicates the start of a sequence. 当提示解码器输出没有初始序列的输出时，我们可以给它一个指示序列开始的值。

    * Combining the two parts results in what is known as an **encoder-decoder**, or a **seq2seq transformer**

* the encoder-decoders as a whole

    The encoder has, in a sense, encoded the sequence. And the decoder, in turn, using this input alongside its usual sequence input, will take a stab at decoding the sequence. 从某种意义上说，编码器对序列进行了编码。 反过来，解码器将使用此输入及其通常的序列输入来尝试解码序列。

    <img src="./image-20231020154258705.png" alt="image-20231020154258705" style="zoom:40%;" />

    Now that we have both the feature vector and an initial generated word, we don't need the encoder anymore. As we have seen before with the decoder, it can act in an auto-regressive manner. The word it has just output can now be used as an input.

    It then uses a combination of the representation and the word it just generated to generate a second word.

    <img src="./image-20231020154501476.png" alt="image-20231020154501476" style="zoom:40%;" />

* When should one use a seq2seq model?

    * seq2seq tasks; many-to-many: translation, summarization
    * Weights are not necessarily shared across the encoder and decoder
    * Input distribution different from output distribution


### Encoder

BERT is a popular encoder-only architecture, which is the most popular model of its kind.

* The encoder outputs exactly on sequence of numbers per input word. (This numerical representation can also be called a "Feature vector", or "Feature tensor")
    * The dimension of that vector is defined by the architecture of the model, for the base BERT model, it is 768.
    * These representations contain the value of a word, but contextualized. 这些表示包含单词的值，但是是上下文化的。One could say that the vector of 768 values holds the "meaning" of that word in the text. ==The self-attention mechanism is used to do this.==
    * The self-attention mechanism relates to different positions (or different words) in a single sequence, in order to compute a representation of that sequence. 自注意力机制涉及单个序列中的不同位置（或不同单词），以便计算该序列的表示。This means that the resulting representation of a word has been affected by other words in the sequence.

<img src="./image-20231019225039119.png" alt="image-20231019225039119" style="zoom:40%;" />

* Why would one use an encoder?

    * Bi-directional: context from the left, and the right
    * Good at extracting vectors that carry meaningful information about a sequence.
    * Sequence classification, question answering, masked language modeling
    * NLU: Natural Language Understanding
    * Encoders can be used as standalone models in wide variety of tasks. 编码器可以在各种任务中用作独立模型。Example of encoders: BERT, RoBERTa, ALBERT

* Masked Language Modeling (MLM)

    * It's the task of predicting a hidden word in a sequence of words.

        <img src="./image-20231019230921263.png" alt="image-20231019230921263" style="zoom:40%;" />

    * This requires a semantic understanding as well as a syntactic understanding.

        Encoders shine in this scenario in particular, as bidirectional information is crucial here. The encoder needs to have a good understanding of the sequence in order to predict to a masked word, as even if the text is grammatically correct, It does not necessarily make sense in the context of the sequence. 编码器在这种情况下尤其出色，因为双向信息在这里至关重要。编码器需要对序列有很好的理解，以便预测屏蔽词，因为即使文本在语法上是正确的，它在序列的上下文中也不一定有意义。

* Sentiment analysis (Analyze the sentiment of a sequence.)

    * Encoders are good at obtaining an understanding of sequences; and the relationship/interdependence between words.

### Decoder

* One can use a decoder for most of the same tasks as an encoder, albeit with, generally, a little loss of performance.

* the input and output is similar to the encoder.

* Where the decoder differs from the encoder is principally with its self-attention mechanism.

    * masked self-attention

        all the words on the right (also known as the right context) of the word is masked.

        the decoders only have access to the words on their left.

        <img src="./image-20231020151320964.png" alt="image-20231020151320964" style="zoom:33%;" />

        the masked self-attention mechanism differs from the self-attention mechanism by using an additional mask to hide the context on either side of the word: the word's numerical representation will not be affected by the words in the hidden context.

* When should one use a decoder?

    * Unidirectional: access to their left (or right) context

    * Great at causal tasks; generating sequences

    * NLG: Natural Language generation

        <img src="./image-20231020153015630.png" alt="image-20231020153015630" style="zoom:50%;" />

    * Example of decoders: GPT-2, GPT Neo

## 2 Attention

### 2.1 Scaled Dot-Product Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vector. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

<img src="./image-20231005194949601.png" alt="image-20231005194949601" style="zoom:70%;" />
$$
Attention(Q, K, V) j= softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

$Q$: a set of queries

$K$: keys of dimension $d_k$

$V$: values of dimension $d_v$

```python
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)  # 取最后一维的size
    # 转置key的后两维后，如上方公式所示，进行矩阵相乘，并做除法
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None: # 用于区分encoder和decoder的attention
        scores = scores.masked_fill(mask == 0, -1e9)  # 将scores中值为0的数填充为-1e9
    p_attn = scores.softmax(dim=-1)  # 为最后一维做softmax
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

Dot-product attention is much faster and more space-efficient than additive attention in practice, since it can be implemented used highly optimized matrix multiplication code.

For large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where the has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

### 2.2 Multi-Head Attention

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

<img src="./image-20231005211628145.png" alt="image-20231005211628145" style="zoom:60%;" />
$$
MultiHead(Q, K, V) = Concat(head_1\;,...,\;head_h)W^O where\;head_i\\
= Attention(QW_i^Q, KW_i^K, VW_i^V)
$$
Where the projections are parameter matrices $W_i^Q \in \mathbb{R}^{model \times d_k},\;W_i^K \in \mathbb{R}^{model \times d_k},\;W_i^V \in \mathbb{R}^{model \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_model}$.

In this work we are employ $h = 8$ parallel attention layers, or heads. For each of these we use $d_k = d_v = d_{model}/h = 64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads"""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # clones为自定义函数，用于产生N个相同的layers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 在mask的第1维（下标从0开始）上增加一个维度
        nbatches = query.size(0)  # batch的数量等于query的第0维

        # 1) Do all the linear projection in batch from d_model => h x d_k
        query, key, value = [
            # 一层一层，将一维的lin(x)投射到 nbatches x h d_k 四维，再转置
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)  # 转置x的1，2维度
            .contiguous()  # 返回一个连续的内存张量，其中包含与 self 张量相同的数据。如果 self 张量已经是指定的内存格式，则此函数返回 self 张量。
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
```

## 2.3 Applications of Attention in our Model

* In "encoder-decoder attention" layers.
    * The queries come from the previous decoder layer
    * The memory keys and values come from the output of the encoder.
    * This allow every position in the decoder to attend over all positions in the input sequence. 这使得decoder中的每个位置都能关注输入序列中的所有位置。
* Self-attention layers in encoder
    * All of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.
    * Each position in the encoder can attend to all positions in the previous layer of the encoder. encoder中的所有位置都可以关注encoder上一层的所有位置。

* Self-attention in decoder
    * Each position in the decoder to attend to all positions in the decoder up to and including that position.
    * We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. 我们需要在decoder中防止信息向左流动，以保持自回归性。
    * We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections.

## 3 其他组件

### 3.1 Position-wise Feed-Forward Networks

Each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. The consists of two linear transformations with a ReLU[^1] activation in between.
$$
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
$$

While the linear transformations are the same across different positions, they use different parameters from layer to layer. The dimensionality of input and output is $d_{model} = 512$, and the inner-layer has dimensionality $d_{ff} = 2048$.

```python
class PositionwiseFeedForward(nn.Module):
    """Implement FFN equation"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
```

### 3.2 Embeddings and Softmax

We used learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{model}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight maxtrix between the two embedding layers and the pro-softmax linear transformation. In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.

### 3.3 Positional Encoding

In order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed.

In this work, we use sine and cosine functions of different frequencies:
$$
PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})
$$
where **pos** is the position and **i** is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths from a geometric progression from $2\pi$ to $1000\;\cdot\;2\pi$. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k_i$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

Sinusoidal version may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

In addition, we apply dropout to the sums of the embeddings and the positional encoding in both the encoder and decoder stacks. For the base model, we use a rate of $P_{drop} = 0.1$.

```python
class PositionalEncoding(nn.Module):
    """Implement the PE function"""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # p[:, 0::2] 从0开始，每隔2个取一个值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # 在内存中定义一个常量，方便写入和写出

    def forward(self, x):
        # requires_grad_(False) 用于让参数不被追踪
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
```

## 4 Encoder and Decoder

The encoder maps an input sequence of symbol representations $(x_1, \dots, x_n)$ to a sequence of continuous representations $\rm{z} = (z_1, \dots, z_n)$. Given $\rm{z}$, the decoder then generates an output sequence $(y_1, \dots, y_m)$ of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbol as additional input when generating the next.

<img src="./image-20231006155825393.png" alt="image-20231006155825393" style="zoom:60%;" />

```python
class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decoder(self.src_embed(src, src_mask), src_mask, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
```



```python
class Generator(nn.Module):
    """Define standard linear + softmax generation step"""

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
```

### 4.1 Encoder

The encoder is composed of a stack of $N = 6$ identical layers.

```python
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn"""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is $LayerNorm(x + Sublayer(x))$, where $Sublayer(x)$ is the function implemented by sub_layer itself. We apply dropout to the each sub-layer, before it is added to the sub-layer input and normalized.

To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{model} = 512$.

```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    请注意，为简化代码，norm 是第一个而不是最后一个。
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size"""
        return x + self.dropout(sublayer(self.norm(x)))
```

Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.

```python
class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (define below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```

### 4.2 Decoder

The decoder is also composed of a stack of $N = 6$ identical layers.

```python
class Decoder(nn.Module):
    """Generic N layer decoder with masking"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```

In addition to the two sub-layers in each layer, **the decoder inserts a third sub-layer**, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.

```python
class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```

We also modify the self-attention sub-layer in the decoder stack to **prevent positions from attending to subsequent positions**. The masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

```python
def subsequent_mask(size):
    """Mark out subsequent positions."""
    attn_shape = (1, size, size)
    # 返回一个张量，包含输入矩阵的上三角部分，其余被设为0
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0
```



[^1]: $ReLU(x) = max(0, x)$
