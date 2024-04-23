# %% imports
import logging
import warnings

import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset, load_dataset
from evaluation.utils import compute_metrics_hf, set_random_seed
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
magpie = "mediabiasgroup/lbm_without_media_bias_pretrained"
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# %% prepare model & data
anno_lex = load_dataset("mediabiasgroup/anno-lexical")


model = AutoModelForSequenceClassification.from_pretrained(
    base_model, num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% tokenize data


tok = tokenizer(
    list(anno_lex["train"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_train_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": anno_lex["train"]["label"],
    },
)

tok = tokenizer(
    list(anno_lex["dev"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_dev_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": anno_lex["dev"]["label"],
    },
)

tok = tokenizer(
    list(anno_lex["test"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt",
)
anno_lex_test_t = Dataset.from_dict(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": anno_lex["test"]["label"],
    },
)


def train_wrapper():
    """Execute the wandb hyperparameter tuning job.

    Takes the (globally defined) tasks, instantiates a trainer for them.
    This function is passed as a callback to wandb.
    """
    wandb.init(
        entity="media-bias-group", project="annomatic_dataset_hyperparams"
    )
    set_random_seed()
    sweep_model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2
    )

    training_args = TrainingArguments(
        report_to="wandb",
        output_dir="./checkpoints",
        per_device_eval_batch_size=64,
        per_device_train_batch_size=32,
        num_train_epochs=wandb.config.epoch,
        evaluation_strategy="steps",
        logging_steps=25,
        eval_steps=25,
        disable_tqdm=False,
        weight_decay=wandb.config.weight_decay,
        warmup_ratio=0.1,
        learning_rate=wandb.config.lr,
        run_name="hyperparam-tuning",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        sweep_model,
        training_args,
        train_dataset=anno_lex_train_t,
        eval_dataset=anno_lex_dev_t,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_hf,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()


# wandb sweep

# first search
# hyper_param_dict = {
#     "epoch": {"values": [3,5,10]},
#     "lr": {"values": [2e-5, 3e-5, 4e-5, 1e-4]},
#     "weight_decay": {"values": [0.01,0.05,0.1]},
# }

# second search
hyper_param_dict = {
    "epoch": {"values": [3]},
    "lr": {"values": [1e-5, 2e-5, 3e-5, 5e-6, 1e-6]},
    "weight_decay": {"values": [0.05]},
}

sweep_config = {"method": "grid"}
sweep_config["parameters"] = hyper_param_dict
sweep_config["name"] = "hyperparam-tuning"
sweep_id = wandb.sweep(sweep_config, project="annomatic_dataset_hyperparams")

wandb.agent(sweep_id, train_wrapper)
