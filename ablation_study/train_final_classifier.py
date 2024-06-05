# %% imports
import logging
import warnings

import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset, load_dataset
from evaluation.utils import compute_metrics, compute_metrics_hf, set_random_seed
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# %% warning surpress
transformers.logging.set_verbosity(transformers.logging.ERROR)
logging.disable(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# %% definitions
base_model = "roberta-base"
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# %% prepare model & data
train = pd.read_parquet('ablation_study/data/anno-lexical-train-left.parquet')
dev = pd.read_parquet('ablation_study/data/anno-lexical-dev-left.parquet')
test = pd.read_parquet('annomatic-dataset/data/training/anno-lexical-test.parquet')

# prep model
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% tokenize data
# train
tok = tokenizer(
    train['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_train_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": train['label'].tolist(),
    },
)
# dev
tok = tokenizer(
    dev['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_dev_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": dev['label'].tolist(),
    },
)

# test
tok = tokenizer(
    test['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_test_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": test['label'].tolist(),
    },
)


# %%
wandb.init(entity="media-bias-group", project="ablation_study")
set_random_seed()

training_args = TrainingArguments(
    report_to="wandb",
    output_dir="./checkpoints",
    per_device_eval_batch_size=32,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    save_total_limit=3,
    evaluation_strategy="steps",
    logging_steps=50,
    eval_steps=50,
    save_steps=50,
    disable_tqdm=False,
    weight_decay=0.05,
    learning_rate=2e-5,
    run_name="left_",
    metric_for_best_model="eval_loss",
    save_strategy="steps",
    load_best_model_at_end=True,
    remove_unused_columns=False,
)

trainer = Trainer(
    model,
    training_args,
    train_dataset=anno_lex_train_t,
    eval_dataset=anno_lex_dev_t,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_hf,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.0,
        ),
    ],
)


trainer.train()
# %%
eval_dataloader = DataLoader(
    anno_lex_test_t,
    batch_size=32,
    collate_fn=data_collator,
)
print(compute_metrics(eval_dataloader, model))

model.push_to_hub("mediabiasgroup/ablation-left")

wandb.log(compute_metrics(eval_dataloader, model))
wandb.finish()
