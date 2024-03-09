from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.training.arguments.data_arguments import DataArguments
from src.training.arguments.model_arguments import ModelArguments
from src.training.arguments.training_option_arguments import TrainingOptionArguments
from src.training.dataset.training_datasets import TrainingDataset
from src.training.utils.metrics import compute_metrics


class TextClassificationTrainer:
    def __init__(
        self,
        data_arguments: DataArguments,
        model_arguments: ModelArguments,
        training_option_arguments: TrainingOptionArguments,
    ):
        self.data_arguments = data_arguments
        self.model_arguments = model_arguments
        self.dataset = self._get_dataset()
        self.training_option_arguments = training_option_arguments

    def _model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_arguments.model_name)

    def _get_dataset(self):
        dataset = TrainingDataset(
            dataset_name=self.data_arguments.dataset_name,
            model_name=self.model_arguments.model_name,
        ).get_dataset()
        return dataset

    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_arguments.model_name)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        training_arguments = TrainingArguments(
            output_dir=self.training_option_arguments.output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.training_option_arguments.batch_size,
            per_device_eval_batch_size=self.training_option_arguments.batch_size,
            fp16=self.training_option_arguments.fp16,
            gradient_accumulation_steps=self.training_option_arguments.gradient_accumulation_steps,
            weight_decay=self.training_option_arguments.weight_decay,
            warmup_ratio=self.training_option_arguments.warmup_ratio,
            learning_rate=self.training_option_arguments.learning_rate,
            num_train_epochs=self.training_option_arguments.num_train_epochs,
            push_to_hub=self.training_option_arguments.push_to_hub,
            seed=self.training_option_arguments.seed,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="f1-score",
            label_smoothing_factor=self.training_option_arguments.label_smoothing_factor,
            report_to=self.training_option_arguments.logger,
        )

        trainer = Trainer(
            model_init=self._model_init,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            data_collator=data_collator,
            args=training_arguments,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
