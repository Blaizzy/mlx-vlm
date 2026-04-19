from .qwen3_dflash import DFlashDraftModel


def load_drafter(path_or_repo: str, kind: str = "dflash", **kwargs):
    if kind != "dflash":
        raise ValueError(f"Unknown drafter kind {kind!r}. Known: ['dflash']")
    from ...utils import get_model_path, load_model

    return load_model(get_model_path(path_or_repo), **kwargs)


__all__ = ["DFlashDraftModel", "load_drafter"]
