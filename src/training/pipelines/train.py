from datasets import load_dataset
from transformers import HfArgumentParser

from src.training.arguments.data_arguments import DataArguments
from src.training.arguments.model_arguments import ModelArguments
from src.training.arguments.training_option_arguments import TrainingOptionArguments
from src.training.trainers.text_classification_trainer import TextClassificationTrainer

if __name__ == "__main__":
    parser = HfArgumentParser((DataArguments, ModelArguments, TrainingOptionArguments))
    data_arguments, model_arguments, training_option_arguments = parser.parse_args_into_dataclasses()

    trainer = TextClassificationTrainer(
        data_arguments=data_arguments,
        model_arguments=model_arguments,
        training_option_arguments=training_option_arguments,
    )

    trainer.train()
