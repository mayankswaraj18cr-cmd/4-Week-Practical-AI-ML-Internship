"""Hugging Face training factory with lazy imports for lightweight use."""

from typing import Any, Dict


def build_text_classifier(model_name: str, num_labels: int = 2) -> Any:
    """Load a sequence classifier; model downloads are performed only on call."""
    if not model_name.strip() or num_labels < 2:
        raise ValueError("model_name must be non-empty and num_labels must be >= 2")
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError("install transformers to use Hugging Face fine-tuning") from exc
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def build_trainer(model: Any, tokenizer: Any, train_dataset: Any, eval_dataset: Any, output_dir: str = "./artifacts") -> Any:
    """Construct a Trainer with conservative, reproducible defaults."""
    try:
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError("install transformers to use the Trainer") from exc
    arguments = TrainingArguments(output_dir=output_dir, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, report_to="none")
    return Trainer(model=model, args=arguments, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer))
