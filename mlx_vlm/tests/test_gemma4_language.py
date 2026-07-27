import mlx.core as mx

from mlx_vlm.models.gemma4.language import LanguageModel


class FakeGemma4TextModel:
    def __call__(self, *args, **kwargs):
        return mx.arange(24).reshape(1, 4, 6)


def test_gemma4_language_model_slices_hidden_before_logits():
    language_model = LanguageModel.__new__(LanguageModel)
    language_model.model = FakeGemma4TextModel()
    language_model.logits_from_hidden = lambda hidden: hidden

    output = language_model(mx.array([[1, 2, 3, 4]]), logits_to_keep=1)

    assert LanguageModel.supports_logits_to_keep is True
    assert output.logits.shape == (1, 1, 6)
    assert mx.array_equal(output.logits, mx.array([[[18, 19, 20, 21, 22, 23]]]))
