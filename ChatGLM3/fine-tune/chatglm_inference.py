#!/usr/bin/env python
# coding: utf-8

# # 模型推理 - 使用 QLoRA 微调后的 ChatGLM-6B

# In[1]:


import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# 模型ID或本地路径
model_name_or_path = '/data/chatglm3-6b'


# In[2]:


_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

# QLoRA 量化配置
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])

# 加载量化后模型(与微调的 revision 保持一致）
base_model = AutoModel.from_pretrained(model_name_or_path,
                                      quantization_config=q_config,
                                      device_map='auto',
                                      trust_remote_code=True,
                                      revision='b098244')


# In[3]:


base_model.requires_grad_(False)
base_model.eval()


# In[4]:


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          trust_remote_code=True,
                                          revision='b098244')


# ## 使用原始 ChatGLM3-6B 模型

# In[5]:


input_text = "解释下乾卦是什么？"


# In[6]:


response, history = base_model.chat(tokenizer, query=input_text)


# In[7]:


print(response)


# #### 询问一个64卦相关问题（应该不在 ChatGLM3-6B 预训练数据中）

# In[8]:


response, history = base_model.chat(tokenizer, query="周易中的讼卦是什么？", history=history)
print(response)


# In[ ]:





# ## 使用微调后的 ChatGLM3-6B

# ### 加载 QLoRA Adapter(Epoch=3, automade-dataset(fixed)) - 请根据训练时间戳修改 timestamp 

# In[10]:


from peft import PeftModel, PeftConfig

epochs = 3
timestamp = "20240407_013445"

peft_model_path = f"models/{model_name_or_path}-epoch{epochs}-{timestamp}"

config = PeftConfig.from_pretrained(peft_model_path)
qlora_model = PeftModel.from_pretrained(base_model, peft_model_path)
training_tag=f"ChatGLM3-6B(Epoch=3, automade-dataset(fixed))-{timestamp}"


# In[11]:


def compare_chatglm_results(query, base_model, qlora_model, training_tag):
    base_response, base_history = base_model.chat(tokenizer, query)

    inputs = tokenizer(query, return_tensors="pt").to(0)
    ft_out = qlora_model.generate(**inputs, max_new_tokens=512)
    ft_response = tokenizer.decode(ft_out[0], skip_special_tokens=True)
    
    print(f"问题：{query}\n\n原始输出：\n{base_response}\n\n\n微调后（{training_tag}）：\n{ft_response}")
    return base_response, ft_response


# ### 微调前后效果对比

# In[12]:


base_response, ft_response = compare_chatglm_results("解释下乾卦是什么？", base_model, qlora_model, training_tag)


# In[13]:


base_response, ft_response = compare_chatglm_results("周易中的讼卦是什么", base_model, qlora_model, training_tag)


# In[14]:


base_response, ft_response = compare_chatglm_results("师卦是什么？", base_model, qlora_model, training_tag)


# In[ ]:





# ## 其他模型（错误数据或训练参数）
# 
# #### 加载 QLoRA Adapter(Epoch=3, automade-dataset)

# In[15]:


from peft import PeftModel, PeftConfig

epochs = 3
peft_model_path = f"models/{model_name_or_path}-epoch{epochs}"

config = PeftConfig.from_pretrained(peft_model_path)
qlora_model_e3 = PeftModel.from_pretrained(base_model, peft_model_path)
training_tag = f"ChatGLM3-6B(Epoch=3, automade-dataset)"


# In[16]:


base_response, ft_response = compare_chatglm_results("解释下乾卦是什么？", base_model, qlora_model_e3, training_tag)


# In[17]:


base_response, ft_response = compare_chatglm_results("地水师卦是什么？", base_model, qlora_model_e3, training_tag)


# In[18]:


base_response, ft_response = compare_chatglm_results("周易中的讼卦是什么", base_model, qlora_model_e3, training_tag)


# In[ ]:





# #### 加载 QLoRA Adapter(Epoch=3, Overfit, handmade-dataset)

# In[21]:


from peft import PeftModel, PeftConfig

epochs = 3
peft_model_path = f"models{model_name_or_path}-epoch{epochs}"

config = PeftConfig.from_pretrained(peft_model_path)
qlora_model_e50_handmade = PeftModel.from_pretrained(base_model, peft_model_path)
training_tag = f"ChatGLM3-6B(Epoch=3, handmade-dataset)"


# In[22]:


base_response, ft_response = compare_chatglm_results("解释下乾卦是什么？", base_model, qlora_model_e50_handmade, training_tag)


# In[23]:


base_response, ft_response = compare_chatglm_results("地水师卦", base_model, qlora_model_e50_handmade, training_tag)


# In[24]:


base_response, ft_response = compare_chatglm_results("天水讼卦", base_model, qlora_model_e50_handmade, training_tag)


# In[ ]:




