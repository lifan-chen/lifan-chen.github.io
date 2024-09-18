---
title: KAPING
categories:
  - Note
  - AI
  - NLP
abbrlink: 61574
date: 2023-12-21 17:04:45
---

Paper: Knowledge-Augmented Language Model Prompting for Zero-Shot Knowledge Graph Question Answering

* 现状
    * LLMs are capable of performing zero-shot closed-book question answering tasks, based on their internal knowledge stored in parameters during pre-training. However, such internalized knowledge might be insufficient, incorrect and out-dated, which could lead LLMs to generate factually wrong answers, known as *hallucination*.
    * Fine-tuning LLMs to update their knowledge is expensive, especially when knowledge is constantly changing.
* 解决方法
    * We propose to retrieve and inject the relevant the knowledge directly as an input, called a *prompt*, to LLMs. Our framework, Knowledge-Augmented language model PromptING (KAPING), requires no model training, thus completely zero-shot.
    * As a knowledge source, we use a Knowledge Graph (KG) consisting of symbolic knowledge in the form of a triple: (head entity, relation, tail entity).
    * We first retrieve the relevant facts to the input question from the knowledge graph based on semantic similarities between the question and its associated facts.
    * After that, triples associated to entities in the KG are verbalized and prepended to the input question, which are then forwarded to LLMs to generate the answer.
    * Consequently, LLMs conditioned on the factual knowledge are able to generate the factual answers, alleviating the hallucination issue, while keeping LLMs' parameters unchanged.
* 上述方法存在的挑战和优化
    * First, most retrieved triples associated with the question entities are unrelated to answer the given question. Therefore, they might mislead the model into generating incorrect answers. 首先，大多数检索到的与问题实体相关的三元组与回答给定的问题无关。
    * On the other hand, the number of triples for the question entities is occasionally large, thereby encoding all triples including unnecessary ones yields high computational costs, especially on LLMs. 另一方面，问题实体的三元组数量偶尔较大，因此对所有三元组(包括不必要的三元组)进行编码会产生较高的计算成本，特别是在LLMs上。
    * We further propose to filter out unnecessary triples based on their semantic similarities to the input question, inspired by the information retrieval. 受信息检索的启发，我们进一步提出根据三元组与输入问题的语义相似度过滤掉不必要的三元组
        * We first represent the question and its associated verbalized triples in the embedding space. 我们首先在嵌入空间中表示疑问句及其相关的动词化三元组。
        * Then, we retrieve the small number of triples whose embeddings are more close to input question's embedding than others. 然后，我们检索少量的三元组，这些三元组的嵌入比其他三元组更接近输入问题的嵌入。

## KAPING framework

* LM Prompting for Zero-Shot QA

    * Zero-Shot Question Answering

        Given an input question $x$, the Question Answering (QA) system returns an answer $y$, where $x$ and $y$ consist of sequences of tokens: $x = [w_1, w_2, \dots, w_{|x|}]$.

        Let $P$ be a QA model based on the generative Language Model (LM), which generates the conditional probability of answer $y$ for question $x$ as follows: $P(y|x)$.

        Then, ==in contrast to supervised learning that trains model $P$ with a set of annotated $(x,y)$ pairs, zero-shot learning does not use any labeled samples and model training.==

    * LM Prompting

        For every input question $x$, we first modify it with a particular instruction template $T$ into a textual srting $x'$ called a *prompt*, as follows: $T:x \rarr x'$.

        Then we forward the prompt $x'$ to the LLM (i.e., $P$), which then generates the answer (i.e., $y$) through $P(y|x')$.

        Note that this LM prompting scheme does not require any additional model parameter updates (i.e., fine-tuning) on the labeled data, thus appropriate for the target zero-shot QA task.

    * Challenges

        * First, LLMs which rely on the knowledge in parameters, are vulnerable from generating the factually incorrect answer, since the knowledge might be inaccurate, and outdated.
        * Also, refining the internalized knowledge with additional parameter updates is expensive, while it is necessary to reflect the wrong and ever growing knowledge.
        * Lastly, which knowledge LLMs memorize and utilize when generating the answer to the question prompt is unclear, which limits their explainability on the outputs.

* Knowledge-Augmented LM Prompting

    * LM Prompting with Knowledge Graphs

        Instead of relying on the knowledge internalized in parameters, we propose to additionally access and inject the knowledge from the external KG, which contains accurate and up-to-date facts helpful to answer the question.

        Formally, a knowledge graph $\cal{G}$ consists of a set of factual triples $\{(s, r, o)\}$, where $s$ and $o$ denote subject and object entities, and $r$ is a specific type of a relation between them.

        Then, for the question prompt $x'$ transformed from the example question $x=$ "Who is the author of Lady Susan?" via the template $T$, we additionally augment its relevant triple: (Lady Susan, written by, Jane Austen), to the LM prompting schema.

        By doing so, LLMs can generate the correct answer with regard to the augmented knowledge from KGs, formalized as follows: $P(y|x',\cal{G})$.

        * Note

            Since we can provide specific and valid facts in KGs to LLMs whenever they exist, our framework can alleviate and outdated knowledge in LLMs, without costly updating their model parameters.

            Furthermore, we can confirm whether LLMs generate answers based on augmented facts, thus improving the explainability of LM prompting.

    * Knowledge Access

        In order to utilize the related facts to the input question, we first extract the entities in the question. For example, for the question "Who is the author of *Lady Susan*?", we extract the entity "Lady Susan". 

        Then, based on the extracted entity, we find its corresponding entity over the KG, whose incident triples then become associated facts to the input question.

        Note that entity matching can be done by existing entity linking techniques.

    * Knowledge Verbalization

        LLMs are working on textual inputs, whereas factual triples are represented over the symbolic graph. Therefore, before injecting the symbolic fact from KGs to LLMs, we first transform the triple consisting of $(s, r, o)$ into its textual string, called verbalization.

        Concatenating the subject, relation, and object texts in the triple, which we observe works well in LM prompting. For example, one triple (Lady Susan, written by, Jane Austen) is used as is: "(Lady Susan, written by, Jane Austen)", for an LLM's input.

    * Knowledge Injection

        Let assume we have a set of $N$ associated triples $k=\{(s_i,r_i,o_i)\}_{i=1}^{N}$ for question $x$.

        Then similar to instruction template $T:x \rarr x'$, we modify $N$ verbalized triples $k$ along with the instruction for the knowledge injection into the knowledge prompt $k'$, as follows: $T:k \rarr k'$.

        One particular template we use for constructing the prompt is that, we first enumerate $N$ verbalized triples line-by-line and then add the specific instruction: "Below are facts in the form of the triple meaning to answer the question.", at the top of the prompt.

        After that, such the knowledge prompt string, $k'$ , is prepended to the question prompt $x'$, and LLMs conditioned by knowledge and question prompts then sequentially generate the answer tokens, formalized as follows: $P(y|[k',x'])$, where $[·]$ denotes concatenation.

* Question-Relevant Knowledge Retrieval

    * Knowledge Retriever

        To overcome those limitations, we further propose to retrieve and augment only the relevant triples to the question.

        For the verbalized triple and the question, we first embed them onto the representation space with off-the-shelf sentence embedding models for text retrieval and then calculate their similarities

        After that, we use only the top-$K$ similar triples, instead of using all $N$ triples, associated to the given question.

## Analyses

* Main Results
    * For zero-shot LM prompting for QA, the knowledge internalized in LLMs is insufficient to generate factual answers, and **it is important to use only the relevant facts**.
    * For tasks that require factual knowledge under low-resource setups, augmenting the knowledge would be beneficial, instead of increasing model sizes to handle the huge volume of knowledge.
* Retriever Results
    * Regarding the number of hops for the candidate triples to retrieve, we observe that, when we increase the hop-size from one to two, the retriever is more likely to retrieve irrelevant triples that does not include answer entities. Therefore, in our experiments, we retrieve knowledge among 1-hop triples of question entities.
    * For zero-shot KGQA, it would be helpful to leverage LLMs to generate answers based on their internalized and external facts, instead of directly search answer entities over KGs.
* Impact of Correct & Incorrect Retrievals
    * When relevant knowledge is augmented, LLMs can contextualize and generate answers accurately.
    * Meanwhile, incorrectly retrieved knowledge makes LLMs condition on irrelevant facts, and generate wrong answers.
* Varying the Amount of Knowledge
    * Some LMs might be distracted by irrelevant triples when their volumes are high, therefore, failing to select and generate the answer entity.
    * Regarding the encoder-decoder model, when the knowledge is augmented to the model, the model tends to generate shorter answers, which can reduce the decoding time. 
    * However, for the decoder-only model (OPT), the more knowledge we augment, the slower the model becomes, because of its auto-regressive characteristic for digesting the input.
* Impact of Orders of Retrieved Triples
    * The OPT (decode-only) model tends to generate the entity located at the first part of the prompt input.
    * Meanwhile, other LLMs can contextualize the entire prompt input, and generate the entity regardless of its position.
* Effectiveness with Entity Linking
* Case Study
    * The LM can generate the output based on the updated facts, which suggests the potential of adapting LMs without costly updating their parameters.
