---
title: VisCPM
categories:
  - Note
  - AI
mathjax: true
abbrlink: 23111
date: 2023-09-14 10:14:26
---

## 1 Title

Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages

## 2 abs

* MPM, an effective training paradigm for training large multimodal (多模态) models in low-resource language

    MPM demonstrates that Multilingual language models can Pivot zero-shot Multimodal learning across languages

    pretrained on English-only image-text data can well generalize to other languages in a zero-shot manner

* open-source codes and model weights at https://github.com/OpenBMB/VisCPM

## 3 Intro

* the multimodal generative capabilities across images and text can be divided into two categories
    * In the field of image-totext generation, multimodal large language models (like GPT-4 [38], LLaVA [32] and InstructBLIP [13]) exhibit remarkable multimodal conversational and reasoning abilities (多模态对话和推理能力) based on images
    * In the field of text-to-image generation, models such as Imagen [45] and Stable Diffusion [43] excel in generating highly realistic and relevant images (高度逼真和相关的图像) based on text prompts
* the paucity of multimodal data resources in non-English languages, the progress of multimodal research in these languages remains hindered
    * To address this challenge, we propose MPM, an effective training paradigm for large multimodal models in non-English languages
    * multilingual learners can effectively align the visual semantics with newly acquired language based on established multimodal and multilingual alignment
    * MPM divides the non-English multimodal learning into two consecutive stages: <mark>multilingual alignment and multimodal alignment</mark>. The former focuses on building a multilingual model, while the latter culminates in a multimodal model spanning multiple languages.
* for multilingual alignment, MPM harnesses a pretrained multilingual large language model (LLM) as the backbone language model
* for the multimodal alignment, MPM trains the visual modules based on the multilingual model exclusively on English image-text pairs to align English and visual semantics.

## 4 Related Work

* Image-to-text Models

    * Traditional image-to-text generation models mainly focus on the task of image caption and visual question answering.

    * Recently, the mainstream of image-to-text has turned to multimodal LLM, focus on rendering LLM capable of interaction with users (LLM能够与用户进行多模态交互).

        * connect the visual module and LLM with perceivers

            VPGTrans [64] explores the transferability of visual modules across LLM. 探索视觉模块跨LLM的可转移性

            LLaVA [32] and Mini-GPT-4 [68] build visual content-related dialog by transferring image captions into conversation data using GPT-4. 通过使用GPT-4将图像描述转换为对话数据来构建与视觉内容相关的对话

            InstructBLIP [13] and M^3^IT [30] incorporate downstream vision-language datasets to construct instruction data. 结合了下游的视觉语言数据集来构建指令数据

* Text-to-image Models

    * In the early stages: generative adversarial networks and auto-regressive generation.
    * Recently, large-scale diffusion-based text-to-image models such as DALLE-2, Imagen, and Stable Diffusion have taken center stage.

* Multilingual Multimodal Models

    * the extension of multimodal models to include multilingual capabilities has become a key research focus over the post few years.

    * make efforts to extend the powerful image-text model CLIP to handle more languages using techniques such as <mark>knowledge distillation [5]</mark> or <mark>contrastive learning [4, 10, 25]</mark>

    * Other studies have aimed to create a universal framework for multilingual vision-language pretraining.

    * Differing from these studies, which try to simultaneously achieve multilingual and multimodal alignment, we focus on effectively leveraging pretrained multilingual LLMs in multimodal learning across various languages.

        * cross-lingual transfer from multilingual LLM in multimodal setting

            Ying-VLM shows that instruction tuning in English can generalize to other languages.

            MultiFusion discover that the multilingual language model can help cross-lingual transfer in text-to-image generation.

    * Differently, our proposed MPM provides a more systematical formulation for the training of multilingual multimodal models and demonstrates that the zero-shot transfer performance of these models can surpass that of models trained on native-language multimodal data.

## 5 Method

### 5.1 MPM Training Paradigm

* Multimodal learning can be formulated as modeling the relationship between images, denoted as $x $, and text, denoted as $y $, in a target language $l_t $.

    we introduce the privot language $l_p $, which contains abundant multimodal pairs $D_p = \{(x_i, y_i^{l_p})\}^M_{i = 1}$, where $M \gg N$. ($N
* is the number of $l_t $'s pairs).

    Imitating the human learning mechanism that can naturally align visual concepts with various learned languages, MPM aims to transfer visual concepts learned in the pivot language to the target language.

* MPM divides the multimodal learning process in target language$l_t$into two consecutive stages

    * multilingual alignment

        For this, MPM aims to establish the cross-lingual alignment for $l_t$and$l_p$. This is achieved by directly leveraging a pretrained multilingual LLM, denoted as $f_\alpha$, which can provide close hidden representations for text pair $y^{l_t}$and$y^{l_p}$with similar semantics. i.e.,$f_\alpha(y^{l_t}) \approx f_\alpha(y^{l_p})$

    * multimodal alignment

        For this, MPM utilize the sufficient multimodal resource $D_p$in the pivot language and optimize the image-to-text objective $p_\theta(y^{l_p}|x)$and text-to-image objective $p_\phi(x|y^{l_p})$.

* In the follow sections, we introduce the training process of **multimodal alignment** stage

    It's worth noting that MPM is agnostic to the specific model architecture and training method (对具体的模型架构和训练方法并不敏感), which enables us to flexibly utilize existing highly effective model architectures and training techniques in each task (这使得我们可以在每个任务中灵活地利用现有的高效模型结构和训练技术).

#### 5.1.1 the image-to-text generation

* can be roughly summarized as generating description for input images, is to learn the conditional distribution $p_{\theta}(y^{l_t}|x)$parameterized by $\theta$

* we incorporate an image encoder module $h_\xi$parameterized by $\xi$(以 $\xi$参数化的图像编码器模块 $h_\xi$) to provide visual feature $z = h_xi(x)$. These visual features **z** are then concatenated with the text embedding (将这些视觉特征z与文本嵌入进行拼接) as input the multilingual LLM.

* MPM's training process for image-to-text generation consists of two sub-stages

    * Multimodal Pretraining

        pretrain the visual module to align it with LLM on a large scale of image-text pairs using the language modeling objective:
        $$
        \mathcal L_1(p_\theta, \mathcal D_p) = - \sum_{i=1}^{M}log \, p_\theta (y_i^{l_p}|h_\xi(x_i))
        $$
        fix the parameters of LLM $(\theta = \{\xi\})$(固定LLM的参数) to prevent the powerful capabilities of LLM from being influenced by short texts in the image-text pairs.

    * Instruction Tuning

        To enhance models' capabilities in following human instructions, we conduct instruction tuning on elaborately curated <mark>multimodal instruction tuning datasets</mark> built by blending the existing multimodal instruction tuning datasets in the pivot language and their translated version in the target language.

        Both the visual module and multilingual LLM are fine-tuned, i.e., $\theta = \{\xi, \alpha\}$, by maximizing the probability of the response.

        we find a <mark>*quasi-zero-shot*</mark> transfer capability of multilingual multimodal models in this scenario.

        If excluding the translated variant in the target language and solely performing instruction tuning using the pivot language, when given an image $x$and a question or an instruction $y_q^{l_t}$in the target language, the resultant model responds accurately though mostly in the pivot language. This can be attributed to the close resemblance between the hidden representation of instructions in two languages provided by the multilingual LLM, i.e., $f_\alpha(y_q^{l_p}) \approx f_\alpha(y_q^{l_t})$<mark>没看懂</mark>

        Since both pretraining and instruction tuning stages employ text components solely in the pivot language, the LLM can understand the question in the target language but cannot calibrate the response in the same language. <mark>没看懂</mark>
        
        To stimulate the model to respond in the target language, MPM <mark>incorporates a small number of translated pairs</mark> in the target language during instruction tuning.

#### 5.1.2 the text-to-image generation

* is to synthesize relevant images given input text prompts, is to learn $p_{\phi}(x|y^{l_t})$parameterized by $\phi$

* adopt a similar architecture with <mark>Stable Diffusion[43]</mark>

    It <mark>incorporates a denoising network $g_\delta$ with a UNet architecture</mark> parameterized by $\delta$ as image decoder to generate images given the input prompt.

    The LLM $f_\sigma$ and image decoder $g_\delta$ are interconnected with <mark>cross-attention mechanism[53]</mark>

    Diffusion models denoise a Gaussian noise into data distribution. The denoise network is optimized to remove the noise of noised image $x_\tau$, conditioned on the hidden states of text input provided by the LLM.

* In this stage, $\phi = \{\delta\}$, i.e., the image decoder is trained to align with frozen with frozen LLM. 将图像解码器进行训练，使其与冻结的LLM对齐

    In this way, when input with the unseen prompt in the target language $y^{l_t}$, the multilingual LLM $f_\sigma$ can inherently provide a representation $f_\sigma(y^{l_t})$ close to the seen representation $f_\sigma(y^{l_p})$ of the pivot language prompt with similar sematics. <mark>懂了又没完全懂</mark>

* the capability of text-to-image in the target language can be seamlessly transferred from the pivot language in a zero-shot fashion. 零样本学习，无缝迁移

### 5.2 VISCPM

use Chinese as the target language and English as the pivot language

CPM-Bee: the Chinese-English bilingual language model serves as the backbone multilingual LLM

* have two variations of the model

    * VISCPM-Chat for image-to-text multimodal conversation
    * VISCPM-Paint for text-to-image synthesis

* Are Chinese Multimodal Datasets Enough To Train A Multimodal Model?

    * the poor quality of the dataset escalates existing data resource shorfalls 数据集质量的低下加剧了现有数据资源的不足

    * translation requires an external machine translation model, and translating a large-scale dataset used in pretraining consumes substantial computing resources. 翻译需要外部的机器翻译模型，而翻译用于预训练的大规模数据集需要消耗大量的计算资源

        And it has marginal improvement on the performance 融入翻译对性能的提升微乎其微

* effectively utilizing the English data to achieve knowledge transfer in <mark>multimodal alignment</mark> is the key to developing a powerful Chinese large multimodal model

#### 5.2.1 VISCPM-Chat

* utilize ==the Muffn architecture [62]== as the image encoder

    Muffin directly leverages a pretrained vision-language model ==BEiT-3 [56]== as an inherent bridge module between vision and language.

* the multimodal pretraining stage

    * the visual module is trained on 100M image-text pairs to align with the frozen LLM
    * train VISCPM-Chat for 180K steps with a batch size of 768 and a learning rate of le-5 using Adam optimizer

* the instruction tuning sub-stage

    * utilize bilingual versions of LLaVA 150K and UniMM-Chat, and the English part of M^3^IT
    * ==*quasi-zero-shot* phenomenon in Chinese introduced==
    * incorporate certain Chinese data by translating LLaVA 150K and UniMM-Chat into Chinese using machine translation
    * fine-tune the image encoder and LLM for 80k steps with a batch size of 64
    * the learning rate and optimizer configurations remain the same as the previous sub-stage

#### 5.2.2 VISCPM-Paint

* ==employs the UNet in Stable Diffusion [43]== as the image decoder

    to maintain the generative capability of UNet, the training process only involves the cross-attention layer of UNet and the linear transblock between the multilingual LLM and UNet. 为了保持UNet的生成能力，训练过程只涉及UNet的交叉注意力层和多语言LLM与UNet之间的线性转换。

* using an extensive dataset of English image-text pairs Laion-2B.

* first train 200K steps at resolution 384 x 384 with a batch-size 4096, and then train 100K steps at resolution 512 x 512 with the same batch size.

    use AdamW optimizer and set the learning rate as 1e-4.

## 6 Exp

* Evaluation of VISCPM-Chat
* Evaluation of VISCPM-Paint
* Ablation Study
* Generalization to More Languages 

## 7 Conclusion

* MPM
* utilize a multilingual LLM as a pivotal intermediary between vision signals and target languages
* VISCPM shows remarkable capability in Chinese image-to-text and text-to-image tasks
* by only rely on English multimodal data

## 8 Source Code

分词器：CPMBee

Beit3
