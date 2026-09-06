"""DINOv2 architecture presets for the official ViT-*/14 (LVD-142M) checkpoints."""

DINOV2_PRESETS = {
    "vits14": dict(embed_dim=384, depth=12, num_heads=6, ffn="mlp"),
    "vitb14": dict(embed_dim=768, depth=12, num_heads=12, ffn="mlp"),
    "vitl14": dict(embed_dim=1024, depth=24, num_heads=16, ffn="mlp"),
    "vitg14": dict(embed_dim=1536, depth=40, num_heads=24, ffn="swiglu"),
}
