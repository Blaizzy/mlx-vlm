from .qwen3_dflash import DFlashDraftModel

KNOWN_DRAFTER_KINDS = {"dflash", "mtp"}


def load_drafter(path_or_repo: str, kind: str = "dflash", **kwargs):
    if kind not in KNOWN_DRAFTER_KINDS:
        raise ValueError(
            f"Unknown drafter kind {kind!r}. Known: {sorted(KNOWN_DRAFTER_KINDS)}"
        )
    from ...utils import get_model_path, load_model

    return load_model(get_model_path(path_or_repo), **kwargs)


__all__ = ["DFlashDraftModel", "KNOWN_DRAFTER_KINDS", "load_drafter"]
