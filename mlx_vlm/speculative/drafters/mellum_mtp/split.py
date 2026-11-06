from ..qwen3_5_mtp.split import Qwen3NextMTPSplitter


class MellumMTPSplitter(Qwen3NextMTPSplitter):
    output_model_type = "mellum_mtp"
    depth_field = "num_nextn_predict_layers"
    block_size_extra = 1

    def read_text_config(self, source_config):
        return dict(source_config.get("text_config") or source_config)

    def run_sanitize(self, tensors, text_config):
        # Unlike Qwen3-Next, Mellum uses ordinary, not zero-centered, RMSNorm.
        return {key.removeprefix("mtp."): value for key, value in tensors.items()}
