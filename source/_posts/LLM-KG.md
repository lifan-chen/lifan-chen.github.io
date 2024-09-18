---
title: LLM+KG
categories:
  - Note
  - AI
abbrlink: 25614
date: 2023-10-28 20:37:42
---

> 中国电子技术标准化研究院：知识图谱与大模型融合实践研究报告
>
> Unifying Large Language Models and Knowledge Graphs: A Roadmap

## 背景

### 知识图谱

Knowledge graphs (KGs) store structured knowledge as a collection of triple $\mathcal{KG} = \{(h, r, t) \subseteq \mathcal{E \times R \times E} \}$, where $\mathcal{E}$ and $\mathcal{R}$ respectively denote the set of entities and relations.

* Categories
    * Encyclopedic Knowledge Graphs 百科类型知识图谱
        * represent the general knowledge in real-world
        * Wikipedia,Knowledge Occean (KO)
    * Commonsense Knowledge Graphs 常识类型知识图谱
        * formulate the knowledge about daily concepts
        * often model the tacit knowledge extracted from text
        * ConceptNet
    * Domain-specific Knowledge Graphs 特定领域知识图谱
        * constructed to represent knowledge in a specific domain
        * often smaller in size, but more accurate and reliable
        * UMLS
    * Multi-modal Knowledge Graphs
        * represent facts in multiple modalities such as images, sounds, and videos
        * can be used for various multi-modal tasks such as image-text matching, visual question answering, and recommendation
        * IMGpedia, MMKG, Richpedia

* 知识图谱与传统知识库相比，具有三大特征
    * 图结构化形式：可诚信为有向图结构化的形式
    * 高效的检索能力：可将概念、实体及其关系结构化组织起来，具有高效检索能力
    * 智能化推理能力：可从已有知识中挖掘和推理更多维的隐含知识
* 落地面临的瓶颈
    * 语料数据标注效率低、主观性强：语料数据标注仍大量依靠人工
    * 知识抽取质量难以保证：知识抽取规则的构建仍主要依赖人工，主观性强，导致可移植性差和误差传播，使得知识抽取质量难以保证
    * 语义理解和自然语言处理难度大：在面对自然语言中的语义歧义、上下文理解、语言常识推理等问题时，仍缺乏有效的解决办法
    * 本体构建难度大：本体构建对领域跨专业知识和构建经验要求高，实体与关系的标识和对齐、本体扩展和更新、本体评估和质控、不同本体融合等方面仍面临技术挑战
    * 知识通用性不足：企业级知识图谱平台及其知识内容具有较强的行业属性和领域专业性，通用性和迁移泛化能力尚有不足，跨行业、跨领域规模化应用有待提升
    * 知识完备性不足：企业级知识图谱构建中通常面临领域边界限制、企业内数据规模有限、数据中知识稀疏等问题，导致其知识完备性不足

### 大模型

* 大模型与传统模型相比具有三大特征
    * 具有涌现能力：在特定任务上，随着模型规模的提升，模型性能突然出现显著提升
    * 参数规模十分庞大：不少于十亿（1B），严格意义上需超过一百亿（10B）
    * 具有通用性：能够仅通过提示、微调适应广泛的下游任务
* 落地面临的瓶颈
    * 训练大模型的成本高
    * 训练数据规模和质量不足：面向特定领域、多应用场景的高质量中文语料规模和质量不足
    * 训练过程可控性差：大模型的黑盒问题使得其推理过程很难得到合理的解释和有效的控制，增加了大模型优化的难度，并限制了其在部分领域的应用
    * 输出的可信度不足：大模型的输出结果是根据概率推理而生成，具有随机性和不稳定性，导致其正确性的验证难度大，难以保证结果的准确可信
    * 输出的安全性不足：大模型的开放性导致其存在信息泄露、数据攻击的风险，影响输出结果的鲁棒性和安全性
    * 知识更新的实时性不足
    * 领域知识的覆盖率不足
    * 社会和伦理问题：大模型的输出可能存在于社会和伦理要求相悖的内容，如：生成内容消极、负面、具有破坏性
* 大模型结构的类别
    * Encoder-only LLMs
        * only use the encoder to encode the sentence and understand the relationships between words
        * The common training paradigm is to predict the mask words in an input sentence. This method is unsupervised and can be trained on the large-scale corpus.
        * require adding an extra prediction head to resolve downstream tasks
        * the most effective for tasks that require understanding the entire sentence, such as text classification and named entity recognition

    * Encoder-decoder LLMs
        * The encoder module is responsible for encoding the input sentence into a hidden-space, and the decoder is used to generate the target output text.
        * The training strategies in encoder-decoder LLMs can be more flexible. For example, T5 is pre-trained by masking and predicting spans of masking words. UL2 unifies several training targets such as different masking spans and masking frequencies.
        * Encoder-decoder LLMs are able to directly resolve tasks that generate sentences based on some context, such as summarization, translation, and question answering.

    * Decoder-only LLMs
        * only adopt the decoder module to generate target output text
        * The training paradigm is to predict the next word in the sentence.
        * Large-scale decoder-only LLMs can generally perform downstream tasks from a few examples or simple instructions, without adding prediction heads or finetuning.


### Prompt Engineering

Prompt engineering is a novel field that focuses on creating and refining prompts to maximize the effectiveness of LLMs across various applications and research areas.

A prompt is a sequence of natural language inputs for LLMs that are specified for the task, such as sentiment classification.

* A prompt could contain several elements
    1. Instruction: a short sentence that instructs the model to perform a specific task
    2. Context: provides the context for the input text or few-shot examples
    3. Input Text: the text that needs to be processed by the model
* Example
    * Chain-of-thought (CoT) prompt enables complex reasoning capabilities throught intermediate reasoning steps. 思维链( CoT )提示通过中间推理步骤实现复杂推理能力
    * Liu et al. incorporate external knowledge to design better knowledge-enhanced prompts. Liu等人[ 65 ]结合外部知识设计了更好的知识增强提示
    * Automatic prompt engineer (APE) proposes an automatic prompt generation method to improve the performance of LLMs. Automatic Prompt Engineer提出了一种自动提示生成方法来提高LLMs的性能

## 知识图谱与大模型融合的可行性

* 技术演化层面

* 技术互补层面

    * 互补关系
        * 知识图谱 -> 大模型
            * 知识图谱能够为通用大模型的工业化应用提供行业领域知识支撑，弥补通用大模型语料里专业领域知识的不足。
            * 利用知识图谱中的知识构建测试集，可对大模型的生成能力进行各方面的评估，降低事实性错误的发生概率。
        * 大模型 -> 知识图谱
            * 大模型可以利用语义理解和生成等能力抽取知识，提高知识抽取的准确性和覆盖率，也可以抽取出隐含的、复杂的、多模态的知识，降低图谱的构建成本
            * 大模型可以利用其语义理解和知识遵循等能力，辅助知识图谱的半自动化构建设计、增加知识的全面性和覆盖度，协助更好的完成知识融合和更新
            * 大模型可以辅助提升知识图谱的输出效果，生成更加合理、连贯、有创新性的内容，例如文本、图像、音频等
    * 融合方向
        * 互补：大模型擅长处理自然语言和模糊知识，知识图谱擅长表示结构化知识并进行推理
        * 互动：
            * 大模型可以从文本中提取知识，从而扩展和丰富知识图谱的内容
            * 知识图谱可以成为为大模型提供结构化知识进行语义补充和生成引导
        * 增强：知识图谱可以提高大模型的语义理解和准确性，而大模型可以为知识图谱提供更丰富的语言知识和生成能力

* 知识库建设层面

    两类知识相互融合的桥梁：**Prompt**

    知识图谱可以利用prompt，参与到大模型的训练前的数据构造，训练中的任务，以及训练后的推理结果的约束生成，提升大模型的性能

    大模型可以通过prompt，来执行相应信息提取以及思维链的推理任务，形式化形成不同的知识，例如三元组、多元组或者事件链条

    | 大模型                                                       | 知识图谱                                                     |
    | :----------------------------------------------------------- | ------------------------------------------------------------ |
    | 动态、概率知识库                                             | 静态知识库                                                   |
    | 参数化知识库，通过网络参数存储知识，不易理解                 | 形式化知识库，通过三元组存储知识，结构清晰，查询简单，易于理解 |
    | 隐式知识库，隐式地存储知识，决策过程难归因、解释、溯源       | 显性知识库，显示地存储知识，有助于归因溯源，提高模型行为的可解释性 |
    | 更新难度大，忘记特定的知识更加困难                           | 便于更新、修改、迁移知识                                     |
    | 知识的通用性更强，适合于高通用的知识密度，高专业知识密度（专业语料少）的应用场景 | 知识的领域性更强，适合于高专业知识密度，低通用知识场景       |
    | 具有上下文感知能力、深层语义表示能力和少样本学习能力         | 图结构表达能力强                                             |
    | 多模态内容采用模型参数存储，有语义对齐和不可解释性           | 多模态按照知识表示形式存储                                   |

## 知识图谱与大模型融合的技术路径

### 大模型赋能知识图谱的技术路线

### 知识图谱赋能大模型的技术路线

* 应用场景实现示例

    1. 基于大模型增强的知识抽取示例

        [DeepKE: DeepKE是由浙江大学知识图谱团队维护开源知识图谱抽取工具集。 - Gitee.com](https://gitee.com/openkg/deepke/tree/main/example/llm)

    2. 基于知识图谱增强大模型的文档问答

* 关键技术示例

    1. 增强大模型的预训练

        ERNIE: Enhanced Language Representation with Informative Entities

        Knowledge-Aware Language Model Pretraining

        * 利用图结构将知识图谱注入到大模型的输入中，增强大模型预训练能力
        * 通过附加的融合模块将知识图谱注入到大模型，在预训练模型中，可以设计额外的副主任务，然后通过辅助任务对预训练模型加约束来增强大模型预训练能力
        * 知识图谱的链式关系输入到大模型中，作为大模型的预训练语料

    2. 增强大模型的监督微调/对齐微调
        * 通过指令微调训练和基于知识图谱反馈的强化学习
        * 通过文本-知识对齐将知识图谱信息注入到大模型的训练目标中，增强大模型预训练能力

    3. 增强大模型的常识和领域知识推理能力

        Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

        * 将知识图谱当作一个准确的知识库，作为一个外部检索的知识源，提升常识和领域知识方面的推理能力
        * 利用知识图谱的多跳路径作为大模型的输入，提升模型的专业性、可信性、真实性与可解释性
        * 利用知识模型根据问题生成知识综述

    4. 大模型推理的可解释性

        Language models as knowledge bases

        KagNet: Knowledgeaware graph networks for commonsense reasoning

        * 通过动态知识图谱融合增强大模型的推理能力，从知识图谱中较容易地获得一系列规则，即从数据中总结得出的特征子图，作为COT、TOT的指令
        * 通过检索增强知识融合增强大模型的推理能力，知识图谱是典型的图结构，有大量的路径。借助这种链式关系可提升大模型推理的可解释性。
        * 利用大语言模型对基于知识图谱生成的问题进行预测，验证大模型的可解释性

    5. 增强大模型生成结果的评估与验证
        * 把知识图谱当作一个准确的知识库，作为外部检索的知识源，解决事实准确性的问题，并进行事实准确性评估

    6. 增强大模型的对话应用能力
        * 在智能对话等任务中，通过引入知识图谱中的知识对大模型的输出进行约束，提升对话内容的有效性和实时性

    7. 增强大模型的语义理解能力
        * 基于知识图谱，对大模型在输入文本的语义识别过程中进行实习别称补全、实体上下位推理等，提升大模型语义识别的准确性和完整性

    8. 增强大模型的知识溯源能力
        * 基于知识图谱，记录大模型获取知识点和关键数据来源信息及转化路径，并在内容生成或推理时进行完整呈现，便于使用者评估可信度

    9. 增强大模型的知识管理和更新能力
        * 基于知识图谱，对大模型输出中所依赖的动态数据、隐私数据、事实性知识进行统一的管理，并依托知识图谱的知识编辑能力，保障图谱内知识的实时性和正确性

    10. 增强大模型的知识存储和共享能力
        * 基于知识图谱，获取和存储大模型中的隐性知识，并通过知识图谱文件或知识图谱间知识共享协议实现知识交换与流通，提升大模型的知识共享能力

* Roadmap

    * KG-enhanced LLM pre-training: inject knowledge into LLMs during the pre-training stage
        * Integrating KGs into training objective
        
            

        * Integrating KGs into LLM inputs
        
        * Integrating KGs by fusion modules
        
    * KG-enhanced LLM inference: enable LLMs to consider the latest knowledge while generating sentences
        * Dynamic knowledge fusion 动态知识融合
        * ==Retrieval-augmented knowledge fusion 检索增强型知识融合==
            * RAG proposes to combine non-parametric and parametric modules to handle the external knowledge. Given the input text, RAG first searches for relavant KG in the non-parametric module via MIPS to obtain several documents. RAG then treats these documents as hidden variables $z$ and feeds them into the output generator, empowered by Seq2Seq LLMs, as additional context information.
            * Story-fragments further improves architecture by adding an additional module to determine salient knowledge entities and fuse them into the generator to improve the quality of generated long stories.
            * EMAT further improves the efficiency of such a system by encoding external knowledge into a key-value memory and exploiting the fast maximum inner product search for memory querying.
            * PEALM proposes a novel knowledge retriever to help the model to retrieve and attend over documents from a large corpus during the pre-training stage and successfully improves the performance of open-domain question answering.
            * KGLM selects the facts from a knowledge graph using the current context to generate factual sentences. With the help of an external knowledge graph, KGLM could describe facts using out-of-domain words or phrases.
        
    * KG-enhanced LLM interpretability: improve the interpretability of LLMs by using KGs
        * KGs for LLM probing
        * KGs for LLM analysis


### 知识图谱与大模型协同应用的技术路径



