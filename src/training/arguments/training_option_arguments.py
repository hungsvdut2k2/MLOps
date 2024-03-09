from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingOptionArguments:
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory of model"},
    )
    batch_size: Optional[str] = field(default=1, metadata={"help": "Batch size for training and evaluate process"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "Option for using mixed precision training"})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "Value for weight decay"})
    warmup_ratio: Optional[float] = field(default=0.1, metadata={"help": "Ratio for warm up"})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "Learning rate for training process"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "Number of training epochs"})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Option for pushing model to HuggingFace Hub"})
    seed: Optional[int] = field(default=42, metadata={"help": "Random seed"})
    label_smoothing_factor: Optional[float] = field(default=0.1, metadata={"help": "Value of label smoothing factor"})
    logger: Optional[str] = field(
        default="none",
        metadata={"help": "Option for log training steps to Wandb or None"},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "Number of gradient accumulation steps"}
    )
