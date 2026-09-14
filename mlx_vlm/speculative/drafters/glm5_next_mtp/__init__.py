from .config import Glm5NextMTPConfig as ModelConfig
from .config import TextConfig
from .glm5_next_mtp import Glm5NextMTPDraftModel
from .glm5_next_mtp import Glm5NextMTPDraftModel as Model

__all__ = ["Glm5NextMTPDraftModel", "Model", "ModelConfig", "TextConfig"]
