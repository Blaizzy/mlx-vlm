from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ServerRuntime:
    model_cache: dict = field(default_factory=dict)
    response_generator: Optional[Any] = None
    apc_manager: Optional[Any] = None
    metrics: Optional[Any] = None


runtime = ServerRuntime()
