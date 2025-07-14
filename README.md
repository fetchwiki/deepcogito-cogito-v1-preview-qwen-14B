---
license: apache-2.0
library_name: transformers
base_model:
- Qwen/Qwen2.5-14B
pipeline_tag: text-generation
---

<p align="center">
  <img src="images/deep-cogito-logo.png" alt="Logo" width="40%">
</p>


# Cogito v1 preview - 14B

[Blog Post](https://www.deepcogito.com/research/cogito-v1-preview)

The Cogito LLMs are instruction tuned generative models (text in/text out). All models are released under an open license for commercial use.

- Cogito models are hybrid reasoning models. Each model can answer directly (standard LLM), or self-reflect before answering (like reasoning models).
- The LLMs are trained using **Iterated Distillation and Amplification (IDA)** - an scalable and efficient alignment strategy for superintelligence using iterative self-improvement.
- The models have been optimized for coding, STEM, instruction following and general helpfulness, and have significantly higher multilingual, coding and tool calling capabilities than size equivalent counterparts.
  - In both standard and reasoning modes, Cogito v1-preview models outperform their size equivalent counterparts on common industry benchmarks. 
- Each model is trained in over 30 languages and supports a context length of 128k.

# Evaluations
We compare our models against the state of the art size equivalent models in direct mode as well as the reasoning mode. For the direct mode, we compare against Llama / Qwen instruct counterparts. For reasoning, we use Deepseek's R1 distilled counterparts / Qwen's QwQ model.

<p align="left">
  <img src="images/14b_benchmarks.png" alt="Logo" width="90%">
</p>

**Livebench Global Average:**
<p align="left">
  <img src="images/livebench_global_average.png" alt="Logo" width="80%">
</p>

For detailed evaluations, please refer to the [Blog Post](https://www.deepcogito.com/research/cogito-v1-preview).


# Usage
Here is a snippet below for usage with Transformers:

```python
import transformers
import torch

model_id = "deepcogito/cogito-v1-preview-qwen-14B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Give me a short introduction to LLMs."},
]

outputs = pipeline(
    messages,
    max_new_tokens=512,
)

print(outputs[0]["generated_text"][-1])
```



## Implementing extended thinking
- By default, the model will answer in the standard mode. 
- To enable thinking, you can do any one of the two methods:
  - Add a specific system prompt, or 
  - Set `enable_thinking=True` while applying the chat template.


### Method 1 - Add a specific system prompt.
To enable thinking, simply use this in the system prompt `system_instruction = 'Enable deep thinking subroutine.'`

If you already have a system_instruction, then use `system_instruction = 'Enable deep thinking subroutine.' + '\n\n' + system_instruction`.

Here is an example - 

```python
import transformers
import torch

model_id = "deepcogito/cogito-v1-preview-qwen-14B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

DEEP_THINKING_INSTRUCTION = "Enable deep thinking subroutine."

messages = [
    {"role": "system", "content": DEEP_THINKING_INSTRUCTION},
    {"role": "user", "content": "Write a bash script that takes a matrix represented as a string with format '[1,2],[3,4],[5,6]' and prints the transpose in the same format."},
]

outputs = pipeline(
    messages,
    max_new_tokens=512,
)

print(outputs[0]["generated_text"][-1])
```


Similarly, if you have a system prompt, you can append the `DEEP_THINKING_INSTRUCTION` to the beginning in this way - 

```python
DEEP_THINKING_INSTRUCTION = "Enable deep thinking subroutine."

system_prompt = "Reply to each prompt with only the actual code - no explanations."
prompt = "Write a bash script that takes a matrix represented as a string with format '[1,2],[3,4],[5,6]' and prints the transpose in the same format."

messages = [
    {"role": "system", "content": DEEP_THINKING_INSTRUCTION + '\n\n' + system_prompt},
    {"role": "user", "content": prompt}
]
```

### Method 2 - Set enable_thinking=True in the tokenizer
If you are using Huggingface tokenizers, then you can simply use add the argument `enable_thinking=True` to the tokenization (this option is added to the chat template).

Here is an example - 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "deepcogito/cogito-v1-preview-qwen-14B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to LLMs."
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

# Tool Calling
Cogito models support tool calling (single, parallel, multiple and parallel_multiple) both in standard and extended thinking mode.

Here is a snippet -

```python
# First, define a tool
def get_current_temperature(location: str) -> float:
    """
    Get the current temperature at a location.
    
    Args:
        location: The location to get the temperature for, in the format "City, Country"
    Returns:
        The current temperature at the specified location in the specified units, as a float.
    """
    return 22.  # A real function should probably actually get the temperature!

# Next, create a chat and apply the chat template
messages = [
  {"role": "user", "content": "Hey, what's the temperature in Paris right now?"}
]

model_inputs = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], add_generation_prompt=True)

text = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
output_text = tokenizer.batch_decode(outputs)[0][len(text):]
print(output_text)
```

This will result in the output - 
```
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "Paris, France"}}
</tool_call><|im_end|>
```

You can then generate text from this input as normal. If the model generates a tool call, you should add it to the chat like so:

```python
tool_call = {"name": "get_current_temperature", "arguments": {"location": "Paris, France"}}
messages.append({"role": "assistant", "tool_calls": [{"type": "function", "function": tool_call}]})
```

and then call the tool and append the result, with the `tool` role, like so:

```python
messages.append({"role": "tool", "name": "get_current_temperature", "content": "22.0"})
```

After that, you can `generate()` again to let the model use the tool result in the chat:

```python
text = tokenizer.apply_chat_template(messages, tools=[get_current_temperature], add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
output_text = tokenizer.batch_decode(outputs)[0][len(text):]
```

This should result in the string -
```
'The current temperature in Paris is 22.0 degrees.<|im_end|>'
```

## License
This repository and the model weights are licensed under the Apache 2.0 License Agreement.

## Contact
If you would like to reach out to our team, send an email to [contact@deepcogito.com](contact@deepcogito.com).

## Model Information

### Architecture

- **Architecture**: Qwen2ForCausalLM
- **Model Type**: qwen2
- **Vocabulary Size**: 151,665
- **Hidden Size**: 5,120
- **Number of Layers**: 48
- **Attention Heads**: 40

### Model Files

This repository contains metadata about the model files. The actual model files are stored on HuggingFace.

| File | Size | Details |
|------|------|---------|
| [model-00001-of-00006.safetensors](https://huggingface.co/deepcogito/cogito-v1-preview-qwen-14B/blob/main/model-00001-of-00006.safetensors) | 4GB | - |
| [model-00002-of-00006.safetensors](https://huggingface.co/deepcogito/cogito-v1-preview-qwen-14B/blob/main/model-00002-of-00006.safetensors) | 4GB | - |
| [model-00003-of-00006.safetensors](https://huggingface.co/deepcogito/cogito-v1-preview-qwen-14B/blob/main/model-00003-of-00006.safetensors) | 4GB | - |
| [model-00004-of-00006.safetensors](https://huggingface.co/deepcogito/cogito-v1-preview-qwen-14B/blob/main/model-00004-of-00006.safetensors) | 4GB | - |
| [model-00005-of-00006.safetensors](https://huggingface.co/deepcogito/cogito-v1-preview-qwen-14B/blob/main/model-00005-of-00006.safetensors) | 4GB | - |
| [model-00006-of-00006.safetensors](https://huggingface.co/deepcogito/cogito-v1-preview-qwen-14B/blob/main/model-00006-of-00006.safetensors) | 4GB | - |

**Total files**: 6
