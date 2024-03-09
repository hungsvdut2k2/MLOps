from typing import List, Optional

from datasets import load_dataset
from transformers import AutoTokenizer


class TrainingDataset:
    def __init__(self, dataset_name: Optional[str], model_name: Optional[str]) -> None:
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_dataset(self):
        dataset = load_dataset("glue", self.dataset_name)
        dataset = dataset.map(self._normalize_text, batched=True)
        dataset = dataset.map(self._tokenize_text, batched=True, remove_columns=["sentence", "idx"])
        return dataset

    def _normalize_text(self, dataset_rows: List[dict]):
        if "sentence" not in dataset_rows.keys():
            raise ValueError("Dataset must have sentence column")
        texts = dataset_rows["sentence"]
        normalized_text = [text.lower() for text in texts]
        return {"sentence": normalized_text}

    def _tokenize_text(self, dataset_rows: List[dict]):
        if "sentence" not in dataset_rows.keys():
            raise ValueError("Dataset must have sentence column")
        texts = dataset_rows["sentence"]
        tokenized_text = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.max_len_single_sentence,
        )
        return tokenized_text
