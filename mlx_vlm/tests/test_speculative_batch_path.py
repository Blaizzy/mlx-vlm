import mlx_vlm.server.generation as generation


def test_batch_path_is_opt_in(monkeypatch):
    monkeypatch.delenv("MLX_VLM_SPECULATIVE_BATCH", raising=False)
    assert generation._speculative_batch_path_enabled() is False


def test_batch_path_accepts_the_usual_truthy_spellings(monkeypatch):
    for value in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("MLX_VLM_SPECULATIVE_BATCH", value)
        assert generation._speculative_batch_path_enabled() is True


def test_batch_path_stays_off_for_other_values(monkeypatch):
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("MLX_VLM_SPECULATIVE_BATCH", value)
        assert generation._speculative_batch_path_enabled() is False
