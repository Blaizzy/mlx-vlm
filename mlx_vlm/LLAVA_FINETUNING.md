# Fine-tuning LLaVA 1.5 with MLX VLM

This guide walks through fine-tuning **LLaVA 1.5** using LoRA with `mlx_vlm`.

---

## Project Directory Structure
```
project-directory/
│
├── my_images/                  # Folder containing all image files
│   ├── image1.png
│   ├── image2.png
│   └── ...
│
├── split_dataset/              # Folder containing dataset splits
│   ├── train.jsonl
│   ├── test.jsonl
│   └── valid.jsonl
│
├── trained_adapter/            # (Post-training) Stores model artifacts and config files
│
└── output.log                  # Training log file (created after running training)
```
- Place all image files inside the my_images/ folder.
- Place JSONL files for train/test/validation splits in `split_dataset/`.
- Run training commands from the project root directory.
- After training, create `trained_adapter/` and move your resulting artifacts there for clarity and future use.

---

## Dataset Format

- Format: **JSONL** (one training example per line).
- Each entry contains image path and conversational history (for instruction tuning).
- Multiple questions for the *same* image each get their own line.

**Sample:**
```json
{"images": "my_images/image1.png", "messages": [{"role": "user", "content": "What is the city name of the card holder? <image>"}, {"role": "assistant", "content": "Kolkata"}]}
{"images": "my_images/image1.png", "messages": [{"role": "user", "content": "What is the issue date of the ID card? <image>"}, {"role": "assistant", "content": "15 Aug 2025"}]}
```

**Notes:**
- "images": Path to the input image.
- "messages": Conversation between user and assistant.
- Use <image> tag in the user content to denote where the image fits into the prompt.
- **Splitting**: Prepare `train.jsonl`, `test.jsonl`, and `valid.jsonl` splits separately.

## Fine-tuning Command

Launch fine-tuning with:

```sh
python3 -m mlx_vlm.lora \
    --model-path llava-hf/llava-1.5-7b-hf \
    --dataset split_dataset \
    --batch-size 1 \
    --lora-rank 64 \
    --lora-alpha 16 \
    --lora-dropout 0.1 \
    --learning-rate 1e-4 \
    --epochs 2 \
    --print-every 5 2>&1 | tee output.log
```

**Result:**  
After successful execution, the following artifacts will be produced (examples):
- `weights.safetensor`
- `chat_template.jinja`
- `config.json`
- `preprocessor_config.json`
- `processor_config.json`
- `special_tokens_map.json`
- `tokenizer_config.json`
- `tokenizer.json`

Move all resulting files into the `trained_adapter/` directory.

#### Key Arguments Explained:

- `--model-path` → Base model path (llava-hf/llava-1.5-7b-hf).

- `--dataset` → Dataset folder containing train.jsonl, test.jsonl, valid.jsonl.

- `--batch-size` → Set to 1 for large models if memory is limited.

- `--lora-rank`, `--lora-alpha`, `--lora-dropout` → LoRA parameters.

- `--learning-rate` → Typical value 1e-4 to 5e-5.

- `--epochs` → Number of training epochs.

## 5. Validating Your Fine-tuned Model
```sh
python3 -m mlx_vlm.generate \
    --model llava-hf/llava-1.5-7b-hf \
    --adapter-path ./trained_adapter \
    --image ./my_images/image_100.png \
    --prompt "What is the full name on this id card?"
    
```
- Use the `--adapter-path` flag to specify your trained LoRA weights and configs.
- The prompt can correspond to any task your fine-tuning covered (e.g., field extraction, document classification, etc.).
