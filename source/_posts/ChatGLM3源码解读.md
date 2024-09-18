---
title: ChatGLM3源码解读
categories:
  - Note
  - AI
  - NLP
abbrlink: 62234
date: 2023-12-18 18:43:08
---

目前主要解读`chat`和`stream_chat`相关的内容

## 1 chat 方法

```python
# modeling_chatglm.py

class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    ...
    @torch.inference_mode()
    def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
             **kwargs):
        # 若没有历史对话，则初始化history
        if history is None:
            history = []
        # 定义 Logit Processor
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p, "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = tokenizer.build_chat_input(query, history=history, role=role)
        inputs = inputs.to(self.device)
        eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                        tokenizer.get_command("<|observation|>")]
        outputs = self.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
        response = tokenizer.decode(outputs)
        history.append({"role": role, "content": query})
        response, history = self.process_response(response, history)
        return response, history
```

### 1.1. Logits Processor

Logits processor 是在生成过程中，每一个step的score计算完成之后，对score进行进一步的加工，改变模型输出的概率分布，从而影响后续生成结果的处理。

```python
# modeling_chatglm.py

from transformers.generation.logits_process import LogitsProcessor

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
```

使用：

```python
if logits_processor is None:
    logits_processor = LogitsProcessorList()
logits_processor.append(InvalidScoreLogitsProcessor())
gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,"temperature": temperature, "logits_processor": logits_processor, **kwargs}

outputs = self.generate(**inputs, **gen_kwargs, eos_token_id=eos_token_id)
```

### 1.2 tokenizer.build_chat_input

```python
# tokenization_chatglm.py

class ChatGLMTokenizer(PreTrainedTokenizer):
    def get_command(self, token):
        if token in self.special_tokens:
            return self.special_tokens[token]
        assert token in self.tokenizer.special_tokens, f"{token} is not a special token for {self.name}"
        return self.tokenizer.special_tokens[token]
    
    # 构建单条消息
    def build_single_message(self, role, metadata, message):
        # 若role不在列表里，报错
        assert role in ["system", "user", "assistant", "observation"], role
        # 获取role的tokens
        role_tokens = [self.get_command(f"<|{role}|>")] + self.tokenizer.encode(f"{metadata}\n")
        message_tokens = self.tokenizer.encode(message)
        # 最后得到的tokens是 role_tokens + message_tokens
        tokens = role_tokens + message_tokens
        return tokens
	
    # 构建对话的input
    def build_chat_input(self, query, history=None, role="user"):
        if history is None:
            history = []
            input_ids = []
        # 取出history中的所有conten，加入input
        for item in history:
            content = item["content"]
            # 若调用了工具，则在content中加入工具的信息
            if item["role"] == "system" and "tools" in item:
                content = content + "\n" + json.dumps(item["tools"], indent=4, ensure_ascii=False)
                input_ids.extend(self.build_single_message(item["role"], item.get("metadata", ""), content))
        # 将当前query加入input
        input_ids.extend(self.build_single_message(role, "", query))
        input_ids.extend([self.get_command("<|assistant|>")])
        # batch_encode_plus方法继承自transformers.PreTrainedTokenizerBase类
        return self.batch_encode_plus([input_ids], return_tensors="pt", is_split_into_words=True)
```

* transformers.PreTrainedTokenizerBase.batch_encode_plus
    * param
        * **batch_text_or_text_pairs** (`List[str]`, `List[Tuple[str, str]]`, `List[List[str]]`, `List[Tuple[List[str], List[str]]]`, and for not-fast tokenizers, also `List[List[int]]`, `List[Tuple[List[int], List[int]]]`) — Batch of sequences or pair of sequences to be encoded. This can be a list of string/string-sequences/int-sequences or a list of pair of string/string-sequences/int-sequence (see details in `encode_plus`).
        * **return_tensors** (`str` or [TensorType](https://huggingface.co/docs/transformers/main/en/internal/file_utils#transformers.TensorType), *optional*) — If set, will return tensors instead of list of python integers. Acceptable values are:`'tf'`: Return TensorFlow `tf.constant` objects.`'pt'`: Return PyTorch `torch.Tensor` objects.`'np'`: Return Numpy `np.ndarray` objects.
    * return
        * transformers.BatchEncoding: This class is derived from a python dictionary and can be used as a dictionary. In addition, this class exposes utility methods to map from word/character space to token space.



## 2 stream_chat 方法

```python
# modeling_chatglm.py

class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    ...
    @torch.inference_mode()
    def stream_chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
    past_key_values=None,max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
    logits_processor=None, return_past_key_values=False, **kwargs):
        if history is None:
        	history = []
        if logits_processor is None:
        	logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
        				tokenizer.get_command("<|observation|>")]
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
        	"temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if past_key_values is None:
        	inputs = tokenizer.build_chat_input(query, history=history, role=role)
        else:
        	inputs = tokenizer.build_chat_input(query, role=role)
        inputs = inputs.to(self.device)
        if past_key_values is not None:
        	past_length = past_key_values[0][0].shape[0]
            # 若之前的sequence长度不为None
        	if self.transformer.pre_seq_len is not None:
                # 已经运行过的长度=pre_seq_len
                past_length -= self.transformer.pre_seq_len
                # 位置id从past_length的id开始
                inputs.position_ids += past_length
                attention_mask = inputs.attention_mask
                # 拼接张量
                attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
                inputs['attention_mask'] = attention_mask
        history.append({"role": role, "content": query})
        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                eos_token_id=eos_token_id, return_past_key_values=return_past_key_values,
                **gen_kwargs):
            if return_past_key_values:
            	outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
            response = tokenizer.decode(outputs)
            if response and response[-1] != "�":
            	response, new_history = self.process_response(response, history)
            	if return_past_key_values:
            		yield response, new_history, past_key_values
            	else:
            		yield response, new_history
```

* 对于几个变量的解释（啊啊啊，感觉要重新去看Transformer）

    * eos_token_id

        是序列结束时标记的id。可以选择使用一个列表来设置多个序列结束标记

    * past_key_value

        只有Decoder模型在文本生成过程（训练过程用不上）中才能用到。顾名思义，它存储的是Decoder模型在 t 时刻前输入的token对应的key和value映射，用于减少计算，将input在$W_k$、$W_v$上的映射存储起来，进行下一个词预测时，就可以直接拿过来用了。它包括self_attention和cross_attention对应的key、value映射。

        单个key或者value单元shape：`[batch_size, n_heads, q_len-1, dim_per_head]`

    * past_key_values

        将每一层的past_key_value都存在其中

### 2.1 stream_generation方法

- return
    - inputs_ids
    - past_key_values

### 2.2 process_response方法

```python
# modeling_chatglm.py

class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    # 处理response
    def process_response(self, output, history):
        content = ""
        history = deepcopy(history)
        # 将response字符串中存在的内容一一归类
        for response in output.split("<|assistant|>"):
            if "\n" in response:  # 好像s
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if not metadata.strip():
                content = content.strip()
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                content = content.replace("[[训练时间]]", "2023年")
            else:
                history.append({"role": "assistant", "metadata": metadata, "content": content})
                if history[0]["role"] == "system" and "tools" in history[0]:
                    content = "\n".join(content.split("\n")[1:-1])
                    def tool_call(**kwargs):
                        return kwargs
                    parameters = eval(content)
                    content = {"name": metadata.strip(), "parameters": parameters}
                else:
                    content = {"name": metadata.strip(), "content": content}
        return content, history

```

