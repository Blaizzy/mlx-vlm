from ..qwen3_5_moe.language import LanguageModel as Qwen3_5MoeLanguageModel
from ..qwen3_5_text.language import LanguageModel as Qwen3_5TextLanguageModel


class LanguageModel(Qwen3_5MoeLanguageModel):
    get_rope_index = Qwen3_5TextLanguageModel.get_rope_index
