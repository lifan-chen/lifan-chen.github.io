---
title: DeepKE
tags:
  - Note
  - AI
  - KG
abbrlink: 22699
date: 2023-11-17 18:47:31
---

## 1 Intro

### 1.1 Knowledge Base (KB)

* As Information Extraction (IE) techniques develop fast, many large-scale Knowledge Bases (KBs) have been constructed.
* Those KBs can provide back-end support for knowledge-intensive tasks in real-world applications, such as
    * language understanding (Che et al., 2021)
    * commonsense reasoning 常识推理 (Lin et al., 2019)
    * recommendation systems (Wang et al., 2018)
* However, most KBs are far from complete due to the emerging entities and relations in real-world application.
* Therefor, Knowledge Base Population (KBP) (Ji and Crishman, 2011) has been proposed, which aims to extract knowledge from the text corpus to complete the missing elements in KBs. 
* For this target, IE is an effective technology that can extract entities and relations from raw texts and link them to KBs. (Yan et al., 2021[^1]; Sui et al., 2021[^2])

### 1.2 现有的 IE toolkits

* To date, a few remarkable open-source and long-term maintained IE toolkits have been develop

    |                         Paper                         | Desc                               |
    | :---------------------------------------------------: | ---------------------------------- |
    |              Spacy (Vasiliev, 2020[^3])               | for named entity recognition (NER) |
    |            OpenNRE (Han et al., 2019[^4])             | for relation extraction (RE)       |
    | Stanford OpenIE (Martínez-Rodríguez et al., 2018[^5]) | for open information extraction    |
    |             RESIN (Wen et al., 2021[^6])              | for event extraction               |

    However, there are still several non-trivial issues that hinder the applicability of real-world applications. 然而，仍然存在一些重要的问题阻碍了实际应用的适用性。

    1. There are various important IE tasks, but most existing toolkits only support on task.
    2. Although IE models trained with those tools can achieve promising results, their performance may **degrade dramatically** when there are only a few training instances (没有 few-shot learning 的能力) or in other complex real-world scenarios, such as encountering document-level and multimodal instances (文档级和多模态).

    Therefore, it is necessary to build a knowledge extaction toolkit facilitating the knowledge base population that supports multiple tasks and complicated scenarios: **low-resource**, document-level and **multimodal**.

### 1.3 DeepKE

* an open-source and extensible knowledge extraction toolkit DeepKE, support complicated low-resource, document-level and multimodal scenarious in knowledge base population.

* Implement information extraction tasks in the standard supervised setting and three complicated scenarios: low-resource, document-level and multimodal settings. 支持在标准的监督设置和低资源、文档级别和多模态三种复杂场景下，进行知识抽取任务

    * named entity recognition

    * relation extraction

    * attribute extraction

* With **a unified framework** (for data processing, model training and evaluation), DeepKE allows developers and researchers to **customize** datasets and models to extract information from unstructured data according to their requirements (可根据需求自定义数据集和模型，从非结构化数据中提取信息) without knowing many technical details, writing tedious glue code, or conducting hyper-parameter tuning.

* Not only provide various functional modules and model implementation for different tasks and scenarios but also organize all components by consistent frameworks to maintain sufficient modularity and extensibility. 不仅为不同的任务和场景提供了不同的功能模块和模型实现，而且通过一致的框架将所有组件组织起来，以保持足够的模块化和可扩展性

* release the source code at [Github](https://github.com/zjunlp/DeepKE) with Google Colab tutorials and [Docs](https://zjunlp.github.io/DeepKE/) for beginners. 

* present [an online system](http://deepke.zjukg.cn/) for real-time extraction of various tasks, and [a demo video](http://deepke.zjukg.cn/demo.mp4).

## 2 Core Functions

<img src="./image-20231117214113599.png" alt="image-20231117214113599" style="zoom:60%;" />

### 2.1 Named Entity Recognition

* As an essential task of IE, named entity recognition (NER) picks out the entity mentions and classifies them into pre-defined (==糟了，这跟我想要的有所差距==) semantic categories given plain texts.
* To achieve supervised NER, DeepKE adopts the pre-trained language model (Devlin et al., 2019 似乎是BERT) to encode sentences and make predictions.
* DeepKE also implements NER in the few-shot setting (including in-domain and cross-domain) (Chen et al., 2022a[^7]) and the multimodal setting.

### 2.2 Relation Extraction

* Relation Extraction (RE), a common task in IE for knowledge base population, predicts semantic relations between pairs of entities from unstructured texts.

* To allow users to customize their models, we adopt various models to accomplish standard supervised RE (我们采用不同的模型来实现标准的有监督RE), including CNN, RNN, Capsule, GCN, Transformer and BERT.

* Meanwhile, DeepKE provides **few-shot and document-level support** for RE.

    * For low-resource RE, DeepKE reimplements *KnowPrompt* (Chen et al., 2022b[^8]), a recent well-performed few-shot RE method based on prompt-tuning.

        > Few-shot RE is significant for real-world applications, which enables users to extract relations with only a few labeled instances.

    * For document-level RE, DeepKE reimplements *DocuNet* (Zhang et al., 2021[^9])  to extract inter-sentence relational triples within one document. 
    * RE is also implemented in the multimodal setting.

### 2.3 Attribute Extraction

* Attribute extraction (AE) plays an indispensable role in the knowledge base population. Given a sentence, entities and queried attribute mentions, AE will infer the corresponding attribute type.

## 3 Toolkit Design and Implementation

### 3.1 Data Module

* The data module is designed for preprocessing and loading input data.
* The tokenizer in DeepKE implements tokenization for both English and Chinese.
* Developers can feed their own datasets into the tokenizer and preprocessor through the dataloader to obtain the tokens or image patches.

### 3.2 Model Module

* The model module contains main neural networks leveraged to achieve three core tasks.
* We implement the **BasicModel** class with a unified **model loader** and **saver** to integrate multifarious neural models.

### 3.3 Core Module

* In the core code of DeepKE, **train**, **validate** and **predict** methods are pivotal components.
    * As for **train** method, users can feed the expected parameters (e.g., the model, data, epoch, optimizer, loss function, .etc) into it without writing tedious glue code.
    * The **validate** method is for evaluation. Users can modify the sentences in the configuration for prediction and then utilize the **predict** method to obtain the result.

### 3.4 Framework Module

* The framework module integrates three aforementioned components and different scenarios.
* It supports various functions, including data processing, model construction and model implementation.
* Meanwhile, developers and researchers can customize all hyper-parameters by modifying configuration files as *"\*.yaml"*, from which we apply [Hydra](https://hydra.cc/) to obtain users' configuration. We also offer an off-the-shelf automatic hyperparameter tuning component.

## 4 Toolkit Usage

### 4.1 Single-sentence Supervised Setting

* Every instance in datasets only contain on sentence.
* The datasets of these tasks are all annotated with specific information, such as entity mentions, entity categories, entity offsets, relation types and attributes. ==糟了，这一部分真的比较麻烦，不能到能不能用==

### 4.2 Low-resource Setting (==可能是需要的功能！！！==)

* In real-world scenarios, labeled data may not be sufficient for deep learning models to make predictions for satisfying users' specific demands. ==对对对，说得太对了，就是这样！！！==
* Therefore, DeepKE provides low-resource ==few-shot support== for NER and RE.
    * DeepKE offers a generative framework with prompt-guided attention to achieve in-domain and cross-domain NER. DeepKE提供了一个具有及时引导注意力的生成框架来实现域内和跨域的NER。
    * Meanwhile, DeepKE implements knowledge-informed prompt-tuning with synergistic for few-shot RE. 同时，DeepKE实现了基于协同优化的知识知情提示调整，用于小样本关系抽取。

### 4.3 Document-Level Setting

* Relations between two entities not only emerge in on sentence but appear in different sentences within the whole document.
* DeepKE can extract inter-sentence relations from documents, which predicts an entity-level relation matrix to capture local and global information.

### 4.4 Multimodal Setting

略



[^1]: Hang Yan, Tao Gui, Junqi Dai, Qipeng Guo, Zheng Zhang, and Xipeng Qiu. 2021. A unified generative framework for various NER subtasks. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021, pages 5808–5822. Association for Computational Linguistics.
[^2]: Dianbo Sui, Chenhao Wang, Yubo Chen, Kang Liu, Jun Zhao, and Wei Bi. 2021. Set generation networks for end-to-end knowledge base population. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, pages 9650–9660. Association for Computational Linguistics.
[^3]: Yuli Vasiliev. 2020. Natural Language Processing with Python and SpaCy: A Practical Introduction. No Starch Press.
[^4]: Xu Han, Tianyu Gao, Yuan Yao, Deming Ye, Zhiyuan Liu, and Maosong Sun. 2019. Opennre: An open and extensible toolkit for neural relation extraction. In Proceedings of EMNLP-IJCNLP.
[^5]: José-Lázaro Martínez-Rodríguez, Ivan López-Arévalo, and Ana B. Ríos-Alvarado. 2018. Openie-based approach for knowledge graph construction from text. Expert Syst. Appl., 113:339–355.
[^6]: Haoyang Wen, Ying Lin, Tuan Lai, Xiaoman Pan, Sha Li, Xudong Lin, Ben Zhou, Manling Li, Haoyu Wang, Hongming Zhang, Xiaodong Yu, Alexander Dong, Zhenhailong Wang, Yi Fung, Piyush Mishra, Qing Lyu, Dídac Surís, Brian Chen, Susan Windisch Brown, Martha Palmer, Chris Callison-Burch, Carl Vondrick, Jiawei Han, Dan Roth, Shih-Fu Chang, and Heng Ji. 2021. RESIN: A dockerized schemaguided cross-document cross-lingual cross-media information extraction and event tracking system. In In Proceedings of NAACL-HLT.
[^7]: Xiang Chen, Lei Li, Shumin Deng, Chuanqi Tan, Changliang Xu, Fei Huang, Luo Si, Huajun Chen, and Ningyu Zhang. 2022a. LightNER: A lightweight tuning paradigm for low-resource NER via pluggable prompting. In Proceedings of the 29th International Conference on Computational Linguistics, pages 2374–2387, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.
[^8]: Xiang Chen, Ningyu Zhang, Xin Xie, Shumin Deng, Yunzhi Yao, Chuanqi Tan, Fei Huang, Luo Si, and Huajun Chen. 2022b. Knowprompt: Knowledgeaware prompt-tuning with synergistic optimization for relation extraction. In WWW ’22: The ACM Web Conference 2022, Virtual Event, Lyon, France, April 25 - 29, 2022, pages 2778–2788. ACM.
[^9]: Ningyu Zhang, Xiang Chen, Xin Xie, Shumin Deng, Chuanqi Tan, Mosha Chen, Fei Huang, Luo Si, and Huajun Chen. 2021. Document-level relation extraction as semantic segmentation. In Proceedings of IJCAI, pages 3999–4006. ijcai.org.
