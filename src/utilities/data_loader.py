"""
Data Loader Utility

Handles loading and tokenization of training datasets.
"""

import json
import os
import random
from typing import Any, List

from datasets import Dataset, DatasetDict
from src.utilities.training_logger import Logger
from src.utilities.data_formatter import DataFormatter

SUPPORTED_LEARNING_TYPES = {
    "random-shuffled",
    "blocked-learning",
    "interleaved-learning",
    "curriculum",
    "blocked-curriculum",
    "interleaved-curriculum",
}


def _normalize_metadata(meta: Any) -> dict:
    """Ensure metadata has consistent types for HuggingFace dataset schema.
    cve_references must be List[str] (not list with nulls); other fields unchanged.
    """
    if not meta or not isinstance(meta, dict):
        return {}
    out = dict(meta)
    cr = out.get("cve_references")
    if cr is not None:
        out["cve_references"] = [
            (x if isinstance(x, str) else "") for x in cr
        ]
    return out


def _load_jsonl_normalized(data_file: str) -> List[dict]:
    """Load JSONL and normalize each record so dataset schema is consistent."""
    records = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "metadata" in rec:
                rec["metadata"] = _normalize_metadata(rec["metadata"])
            records.append(rec)
    return records


def parse_difficulty_from_context_length(
    context_length: int,
    easy_context_max: int,
    medium_context_max: int,
) -> str:
    """Map tokenized context length to curriculum difficulty bucket."""
    if context_length <= easy_context_max:
        return "easy"
    if context_length <= medium_context_max:
        return "medium"
    return "hard"


class DataLoader:
    """Data loading utility (SRP: single responsibility for data loading)"""
    
    def __init__(self, tokenizer: Any, use_4bit: bool):
        self.tokenizer = tokenizer
        self.use_4bit = use_4bit
        self._easy_context_max = None
        self._medium_context_max = None
        self._max_seq_length = None
        # Disable multiprocessing for CUDA compatibility
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def load_and_tokenize(
        self,
        data_file: str,
        easy_context_max: int,
        medium_context_max: int,
        learning_type: str,
        max_seq_length: int = None,
    ) -> DatasetDict:
        """Load and tokenize training dataset"""
        Logger.print_section(f"Loading training data from {data_file}")
        self._easy_context_max = easy_context_max
        self._medium_context_max = medium_context_max
        self._max_seq_length = max_seq_length
        
        records = _load_jsonl_normalized(data_file)
        dataset = Dataset.from_list(records)
        dataset = dataset.add_column("datapoint_id", list(range(len(dataset))))
        Logger.print_info("Loaded examples", len(dataset))
        
        # Split into train/validation
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        Logger.print_info("Train", f"{len(dataset['train'])} | Validation: {len(dataset['test'])}")
        
        Logger.print_info("Tokenizing (using small batches to save memory)", "")
        tokenized_datasets = dataset.map(
            self._tokenize_function,
            batched=True,
            batch_size=10,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing",
            num_proc=None,
            load_from_cache_file=False
        )
        tokenized_datasets["train"] = self._apply_learning_order(
            tokenized_datasets["train"],
            easy_context_max=easy_context_max,
            medium_context_max=medium_context_max,
            learning_type=learning_type,
        )
        if "datapoint_id" in tokenized_datasets["test"].column_names:
            tokenized_datasets["test"] = tokenized_datasets["test"].remove_columns(["datapoint_id"])
        for split_name in ("train", "test"):
            removable = [
                col for col in ("context_length", "difficulty")
                if col in tokenized_datasets[split_name].column_names
            ]
            if removable:
                tokenized_datasets[split_name] = tokenized_datasets[split_name].remove_columns(removable)
        
        Logger.print_success("Data preparation complete!")
        return tokenized_datasets
    
    def _tokenize_function(self, examples):
        """Tokenize training examples"""
        # CRITICAL UPDATE: Pass self.tokenizer to the formatter so it applies the correct model template
        texts = [DataFormatter.format_training_data(ex, tokenizer=self.tokenizer) for ex in examples["messages"]]
        
        max_length = self._max_seq_length if self._max_seq_length is not None else (1024 if self.use_4bit else 2048)
        tokenized = self.tokenizer(
            text=texts, # Change this line
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        tokenized["context_length"] = [
            int(sum(mask)) for mask in tokenized["attention_mask"]
        ]
        tokenized["difficulty"] = [
            parse_difficulty_from_context_length(
                cl,
                easy_context_max=self._easy_context_max,
                medium_context_max=self._medium_context_max,
            )
            for cl in tokenized["context_length"]
        ]
        if "datapoint_id" in examples:
            tokenized["datapoint_id"] = examples["datapoint_id"]
        return tokenized

    def _apply_learning_order(
        self,
        train_split: Dataset,
        easy_context_max: int,
        medium_context_max: int,
        learning_type: str,
    ) -> Dataset:
        """Apply train ordering strategy based on learning type."""
        if "context_length" not in train_split.column_names:
            Logger.print_warning("Learning order skipped", "context_length not found in train split")
            return train_split

        strategy = (learning_type or "").lower()
        if strategy not in SUPPORTED_LEARNING_TYPES:
            Logger.print_warning("Learning type", f"Unknown '{learning_type}', falling back to 'curriculum'")
            strategy = "curriculum"

        if strategy == "curriculum":
            ordered = train_split.sort("context_length")
        elif strategy == "random-shuffled":
            rows = list(train_split)
            rng = random.Random(42)
            rng.shuffle(rows)
            ordered = Dataset.from_list(rows)
        else:
            ordered = self._order_by_bucket_strategy(train_split, learning_type=strategy)

        easy_count = 0
        medium_count = 0
        hard_count = 0
        for level in ordered["difficulty"]:
            if level == "easy":
                easy_count += 1
            elif level == "medium":
                medium_count += 1
            else:
                hard_count += 1
        Logger.print_info("Learning type", strategy)
        Logger.print_info("Difficulty buckets", f"easy: {easy_count}, medium: {medium_count}, hard: {hard_count}")
        Logger.print_info(
            "Curriculum thresholds",
            f"easy<={easy_context_max}, medium<={medium_context_max}, hard>{medium_context_max}",
        )
        return ordered

    def _order_by_bucket_strategy(self, train_split: Dataset, learning_type: str) -> Dataset:
        """Order train split by blocked/interleaved learning or curriculum variants."""
        rows = list(train_split)
        rows_sorted = sorted(rows, key=lambda r: (int(r.get("context_length", 0)), int(r.get("datapoint_id", 0))))
        rows_unsorted = sorted(rows, key=lambda r: int(r.get("datapoint_id", 0)))
        use_sorted_buckets = learning_type in {"blocked-curriculum", "interleaved-curriculum"}
        source_rows = rows_sorted if use_sorted_buckets else rows_unsorted
        easy_rows = [r for r in source_rows if r.get("difficulty") == "easy"]
        medium_rows = [r for r in source_rows if r.get("difficulty") == "medium"]
        hard_rows = [r for r in source_rows if r.get("difficulty") == "hard"]

        if learning_type in {"blocked-learning", "blocked-curriculum"}:
            ordered_rows = easy_rows + medium_rows + hard_rows
            return Dataset.from_list(ordered_rows)

        # interleaved: round-robin easy -> medium -> hard until all buckets are exhausted
        ordered_rows = []
        i = 0
        max_len = max(len(easy_rows), len(medium_rows), len(hard_rows))
        while i < max_len:
            if i < len(easy_rows):
                ordered_rows.append(easy_rows[i])
            if i < len(medium_rows):
                ordered_rows.append(medium_rows[i])
            if i < len(hard_rows):
                ordered_rows.append(hard_rows[i])
            i += 1
        return Dataset.from_list(ordered_rows)
