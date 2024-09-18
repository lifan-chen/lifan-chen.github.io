---
title: Knowledge Graphs Reduce Hallucinations in LLMs
categories:
  - Note
  - AI
  - NLP
abbrlink: 64795
date: 2023-12-20 15:45:41
---

[Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey](https://arxiv.org/abs/2311.07914)

* Knowledge: human acquire discrete facts and skills through learning or experience.

    This acquired knowledge can be interconnected (相互关联) and applied in a logical, multi-step reasoning process to draw conclusions or make judgments, a cognitive function commonly recognized as "*commonsense reasoning*" (常识推理).

Due to the stochastic decoding (随机解码) employed by these LLMs, their behavior is often considered probabilistic.

Unlike a deterministic system providing a single answer for a given input, an LLM generates a spectrum of potential outputs, each associated with a likelihood or probability score. 

These probabilities are grounded in the patterns and associations the LLMs have learned from the extensive training data during the pre-training phase.

This data may also contain **misinformation, biases, or inaccuracies**, leading to the generation of content that reinforces or reflects those biases.

An other challenges is disambiguating certain phrases or terms, as the language is context-dependent.

问题：

* Hallucination: non-existent, unrelated, or wrong answers that are not factually accurate but still sound plausible.

    The hallucinations make these models less dependable.

    The inherent probabilistic nature of these models poses challenges in mitigating hallucination issues. Effectively addressing this concern requires focused research and development efforts in areas such as **continuous knowledge updates, model fine-tuning, and incorporating fact-checking mechanisms (持续知识更新、模型微调以及纳入事实检查机制)** to mitigate the generation of inaccurate or hallucinated content.

    Providing the model with more granular and contextually relevant, relevant, precise external knowledge can significantly aid the model in recalling essential information (显着帮助模型回忆基本信息).

解决思路：

* Research Directions

    * Enhancing LLMs through the augmentation of external knowledge using knowledge representation tools like knowledge graphs.

        * boost the model performance by augmenting comprehensive external knowledge and providing guidance for a more robust reasoning process
        * break down complex, multi-step questions into more manageable sub-questions

    * Using an external information retrieval system to extract knowledge precisely based on the context.

        * explore "chains of thoughts" to unlock the reasoning capabilities of LLMs

            They achieve this by providing intermediate steps through in-context few-shot learning examples via specific prompts.

        * enhance the model's pre-training phase to prevent the inclusion of falsehoods

* Hence, the exploration of augmenting external knowledge through knowledge graphs emerges as promising avenue for advancing the precision, reasoning, and generative capabilities of Large Language Models (LLMs).

    因此，通过知识图增强外部知识的探索成为提高大型语言模型（LLM）的精度、推理和生成能力的有前途的途径。

* Three Overarching Categories of Strategies
    * Knowledge-Aware Inference: refining the inference-making process 细化推理过程
    * Knowledge-Aware Learning: optimizing the learning mechanism 优化学习机制
    * Knowledge-Aware Validation: establishing a means of validating the model's decisions 建立验证模型决策的方法

## Knowledge Graph-Enhanced LLMs

* LLMs primarily have three points of failure 大型语言模型主要存在三个失败点
    * a failure to comprehend the question due to lack of context 由于缺乏上下文而无法理解问题
    * insufficient knowledge to respond accurately 知识不足而无法准确回答
    * an inability to recall specific facts 无法回忆起具体事实
* Improving the cognitive capabilities of these models 提高这些模型的认知能力
    * refining their inference-making process 完善其推理过程
    * optimizing learning mechanisms 优化学习机制
    * establishing a mechanism to validate results 建立验证结果的机制

### Knowledge-Aware Inference

The LLMs face challenges in inferencing, failure to provide correct output, or giving sub-optimal results.

Different factors, such as ambiguous input or lack of clear context, may lead to these failures. The other reason could be a knowledge gap (知识鸿沟), training data biases, or an inability to generalize to new, unseen scenarios.

Unlike humans, LLMs cannot often seek additional information to clarify questions for ambiguous queries and refine their understanding.

Knowledge Graphs (KG) are an excellent source of structured representation of symbolic knowledge of real-world facts. Researchers are actively working on utilizing the existing knowledge graphs and ==augmenting the external knowledge at the input (or prompt level)==, enabling the model to access relevant context and improve its reasoning capabilities.

Classify such techniques further as

* ==KG-Augmented Retrieval==

    Retrieval-augmented generation models like RAG and RALM have gained popularity for enhancing the contextual awareness of large language models (LLMs) when tackling knowledge-intensive tasks.

    They provide relevant documents to the LLMs during the generation process and have effectively mitigated hallucination issues without changing the LLM architecture.

    * the Knowledge-Augmented language model PromptING (KAPING)

        Retrieves the relevant knowledge by matching the entity in question and extracts the related triples from knowledge graphs. These triples were augmented to the prompts for zero-shot question answering.

    * Suggest that rewriting the extracted triples into well-textualized statements further improves the performance of large language models.

    * similarity is insufficient to find relevant facts for complex questions.

        Sen et al. suggested using a retriever module based on a sequence-to-sequence knowledge graph question answering (KGQA) model to predict the distribution over multiple relations in a knowledge graph for question answering. The retrieved *top-k* triples are added as a context to the question given to the LLM.

    * StructGPT uses three structured data sources: knowledge graphs, data tables, and structured databases.

* KG-Augmented Reasoning

* KG-Controlled Generation
