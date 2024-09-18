---
title: LLM
categories:
  - Note
  - AI
  - NLP
  - LLM
abbrlink: 8757
date: 2024-05-08 11:47:19
---

## 1 基础理论

* Few/Zero-Shot Learning
* In-Context Learning
* Chain-of-Thought
* Emergence
* Scaling Prediction
* Parameter-Efficient Learning (Delta Tuning)
* ...

What——大模型学到了什么？

​	大模型的涌现现象 Wei et al. Emergent Abilities of Large Language Models. TMLR 2022.

How——如何训练好大模型？

​	训练规律 Kaplan et al. Scaling Laws for Neural Language Models. 2020

Why——大模型为什么好？

​	关于大模型各种特性的收集 https://github.com/openbmb/BMPrinciples

## 2 网络架构

Transformer以外的更多可能（下一代基础网络框架模型）

## 3 高效计算

* 训练
* 推理
    * 模型压缩：模型剪枝、知识蒸馏、参数量化

## 4 高效适配

* 提示学习

    [1] Tom Brown et al. Language Models are Few-shot Learners. 2020.
    [2] Timo Schick et al. Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference. EACL 2021.
    [3] Tianyu Gao et al. Making Pre-trained Language Models Better Few-shot Learners. ACL 2021.

* 参数高效微调

    [4] Ning Ding et al. Parameter-efficient Fine-tuning for Large-scale Pre-trained Language Models. Nature Machine Intelligence. 
    [5] Neil Houlsby et al. Parameter-Efficient Transfer Learning for NLP. ICML 2020.
    [6] Edward Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

    当基础模型规模增长到一定程度，不同参数高效微调方法的性能差距缩小，且性能与全参数微调基本相当

## 5 可控生成

* 指令微调
* 提示工程
* 思维链

## 6 安全伦理

* 安全：容易被植入后门

* 伦理：与人类价值观对齐

    此前研究表明模型越大会变得越有偏见 Lin et al. TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022.

## 7 认知学习

工具学习

## 8 创新应用

生物、法律……

## 9 数据评估

* 数据

    从多模态数据中学习更加开放和复杂的知识

    [1] OpenAI. GPT-4 Technical Report. 2023.
    [2] Driess D, Xia F, Sajjadi M S M, et al. PaLM-E: An embodied multimodal language model[J]. arXiv preprint arXiv:2303.03378, 2023.

* 评估

    * 自动评价

        选择题 Chiang, Wei-Lin et al. Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality. 2023.

    * 模型评价

        更强大的大模型做裁判 Huang, Yuzhen et al. C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. arXiv preprint arXiv:2305.08322, 2023.

    * 人工评价

## 10 易用性









## 参考文献

[1] https://www.zhihu.com/question/595298808
