"""MLX stream lifecycle helpers."""


def clear_mlx_streams() -> None:
    """Release streams owned by the current thread when MLX supports it."""
    import mlx.core as mx

    clear_streams = getattr(mx, "clear_streams", None)
    if clear_streams is not None:
        clear_streams()
