import mlx.nn as nn
from mlx.utils import tree_unflatten


class RMSNorm(nn.RMSNorm):
    """Keep shifted Qwen norm scales in FP32 without promoting activations."""

    def __call__(self, x):
        return super().__call__(x).astype(x.dtype)


def preserve_norm_precision(language_model):
    replacements = []
    for path, module in language_model.named_modules():
        if isinstance(module, nn.RMSNorm):
            norm = RMSNorm(module.weight.shape[0], eps=module.eps)
            norm.weight = module.weight
            replacements.append((path, norm))
    language_model.update_modules(tree_unflatten(replacements))
