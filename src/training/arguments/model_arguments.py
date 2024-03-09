from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the model to use (via the transformers library)."},
    )
