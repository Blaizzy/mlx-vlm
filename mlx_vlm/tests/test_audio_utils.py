import numpy as np

from mlx_vlm.utils import load_audio


def test_load_audio_uses_mlx_audio_io_and_returns_mono_float32(monkeypatch):
    from mlx_audio import audio_io

    calls = []

    def fake_read(file, dtype="float64"):
        calls.append((file, dtype))
        return (
            np.array(
                [
                    [0.0, 1.0],
                    [0.5, -0.5],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            16000,
        )

    monkeypatch.setattr(audio_io, "read", fake_read)

    audio = load_audio("sample.wav", sr=16000)

    assert calls == [("sample.wav", "float32")]
    assert audio.dtype == np.float32
    assert audio.shape == (3,)
    np.testing.assert_allclose(audio, np.array([0.5, 0.0, 0.5], dtype=np.float32))


def test_load_audio_downmixes_stereo_before_resampling(monkeypatch):
    from mlx_audio import audio_io
    from mlx_audio.utils import resample_audio

    rng = np.random.default_rng(0)
    stereo = rng.standard_normal((48000, 2)).astype(np.float32)

    monkeypatch.setattr(audio_io, "read", lambda file, dtype="float64": (stereo, 48000))

    audio = load_audio("stereo_48k.wav", sr=16000)

    expected = np.asarray(resample_audio(stereo.mean(axis=1), 48000, 16000))
    assert audio.dtype == np.float32
    assert audio.shape == (16000,)
    np.testing.assert_allclose(audio, expected, rtol=1e-5, atol=1e-6)
