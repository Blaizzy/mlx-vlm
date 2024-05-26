
import mlx.nn as nn
import mlx.core as mx

def find_all_linear_names(model):
    cls = nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def collate_fn(processor, examples):
    texts = ["answer " + example["question"] for example in examples]
    labels= [example['multiple_choice_answer'] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                        return_tensors="pt", padding="longest",
                        tokenize_newline_separately=False)

    tokens = tokens.to(mx.float16)
    return tokens