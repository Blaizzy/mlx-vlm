# LLaVA-Phi-3-Mini

An example of LLaVA-Phi-3-Mini: Phi-3-Mini combined with CLIP fine-tuned on 1.27M samples in MLX.[^1] LLaVA is
a multimodal model that can generate text given combined image and text inputs.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run

You can use LLaVA to ask questions about images.

For example, using the command line:

```bash
python generate.py \
  --model xtuner/llava-phi-3-mini-hf \
  --image "http://images.cocodataset.org/val2017/000000039769.jpg" \
  --prompt "USER: <image>\nWhat are these?\nASSISTANT:" \
  --max-tokens 128 \
  --temp 0
```

This uses the following image:

![alt text](http://images.cocodataset.org/val2017/000000039769.jpg)
 
And generates the output:

```
These are two cats sleeping on a pink couch.<|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|><|end|>```

You can also use LLaVA-Phi-3-Mini in Python:

```python
from generate import load_model, prepare_inputs, generate_text

processor, model = load_model("xtuner/llava-phi-3-mini-hf")

max_tokens, temperature = 128, 0.0

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_ids, pixel_values = prepare_inputs(processor, image, prompt)

reply = generate_text(
    input_ids, pixel_values, model, processor, max_tokens, temperature
)

print(reply)
```

[^1]:
    Refer to [xtuner project webpage](https://github.com/InternLM/xtuner/) for more
    information.
