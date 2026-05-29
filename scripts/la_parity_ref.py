"""LocateAnything-3B HF reference dump for MLX parity (runs on pc.lan / CUDA).

Replicates the HF image preprocessing in pure torch (torchvision absent),
forces sdpa attention (flash_attn absent), and saves:
  - the exact model inputs (pixel_values [N,C,14,14], image_grid_hws, input_ids)
  - vision_model features (vit_embeds [merged,4608]) and mlp1 output [merged,2048]
  - the slow/AR greedy generation token ids + decoded text
into ~/la_parity_ref.npz so the MLX side can compare on identical inputs.
"""

import json
import math
import os
import urllib.request

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

MODEL = "nvidia/LocateAnything-3B"
IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
PROMPT = "Detect all objects in the image."
OUT = os.path.expanduser("~/la_parity_ref.npz")
META = os.path.expanduser("~/la_parity_ref.json")
PATCH = 14
MERGE = [2, 2]
IN_TOKEN_LIMIT = 25600
DTYPE = torch.float32
DEVICE = "cuda"


def hf_preprocess(image: Image.Image):
    """Pure-torch replica of LocateAnythingImageProcessor (bicubic ceil-resize)."""
    image = image.convert("RGB")
    w, h = image.size
    if (w // PATCH) * (h // PATCH) > IN_TOKEN_LIMIT:
        scale = math.sqrt(IN_TOKEN_LIMIT / ((w // PATCH) * (h // PATCH)))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    new_w, new_h = image.size
    pad_w, pad_h = MERGE[1] * PATCH, MERGE[0] * PATCH
    target_w = math.ceil(new_w / pad_w) * pad_w
    target_h = math.ceil(new_h / pad_h) * pad_h
    if (target_w, target_h) != (new_w, new_h):
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

    arr = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)  # H,W,C
    arr = arr.permute(2, 0, 1)  # C,H,W  (== TF.to_tensor)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    arr = (arr - mean) / std
    C, H, W = arr.shape
    patches = arr.reshape(C, H // PATCH, PATCH, W // PATCH, PATCH)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous().view(-1, C, PATCH, PATCH)
    return patches, (H // PATCH, W // PATCH)


def main():
    img_path = "/tmp/la_parity_img.jpg"
    if not os.path.exists(img_path):
        urllib.request.urlretrieve(IMG_URL, img_path)
    image = Image.open(img_path)
    # LA_SQUARE=<side> -> resize to a square whose grid == pos-emb size (no interp)
    side = os.environ.get("LA_SQUARE")
    if side:
        image = image.convert("RGB").resize(
            (int(side), int(side)), Image.Resampling.BICUBIC
        )

    pixel_values, (gh, gw) = hf_preprocess(image)
    n_img_tokens = (gh * gw) // (MERGE[0] * MERGE[1])
    print(f"grid={gh}x{gw}  patches={pixel_values.shape[0]}  img_tokens={n_img_tokens}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<img>"
        + "<IMG_CONTEXT>" * n_img_tokens
        + "</img>"
        + PROMPT
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    input_ids = tok(prompt, return_tensors="pt").input_ids
    print("input_ids:", input_ids.shape)

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    for sub in (
        cfg,
        getattr(cfg, "text_config", None),
        getattr(cfg, "vision_config", None),
    ):
        if sub is not None:
            sub._attn_implementation = "sdpa"
    model = AutoModel.from_pretrained(
        MODEL,
        config=cfg,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
    ).to(DEVICE)
    model.train(False)  # eval mode (avoid literal eval() call)

    pv = pixel_values.to(DEVICE, DTYPE)
    gh_t = torch.tensor([[gh, gw]], dtype=torch.int32, device=DEVICE)

    with torch.no_grad():
        vit_list = model.extract_feature(pv, gh_t)
        vit_cat = torch.cat(vit_list, dim=0)
        mlp_out = model.mlp1(vit_cat)
    print("vit_embeds:", tuple(vit_cat.shape), "mlp_out:", tuple(mlp_out.shape))

    decoded, new_ids = "", []
    try:
        with torch.no_grad():
            gen = model.generate(
                pixel_values=pv,
                input_ids=input_ids.to(DEVICE),
                image_grid_hws=gh_t,
                tokenizer=tok,
                use_cache=True,
                max_new_tokens=128,
                generation_mode="slow",
            )
        new_ids = gen[0].tolist()[input_ids.shape[1] :]
        decoded = tok.decode(new_ids, skip_special_tokens=False)
        print("SLOW-MODE OUTPUT:", decoded)
    except Exception as e:  # noqa: BLE001
        decoded = f"<<generation failed: {type(e).__name__}: {e}>>"
        print(decoded)

    np.savez(
        OUT,
        input_ids=input_ids.cpu().numpy().astype(np.int64),
        pixel_values=pixel_values.cpu().float().numpy(),
        image_grid_hws=np.array([[gh, gw]], dtype=np.int64),
        vit_embeds=vit_cat.cpu().float().numpy(),
        mlp_out=mlp_out.cpu().float().numpy(),
        gen_new_ids=np.array(new_ids, dtype=np.int64),
    )
    with open(META, "w") as f:
        json.dump(
            {
                "grid": [gh, gw],
                "n_img_tokens": int(n_img_tokens),
                "decoded": decoded,
                "prompt_tokens": int(input_ids.shape[1]),
            },
            f,
            indent=2,
        )
    print("SAVED", OUT)


if __name__ == "__main__":
    main()
