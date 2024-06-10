# %% imports
import logging
import warnings

import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset, load_dataset
from coreset.utils import compute_metrics, set_random_seed, wandb_run
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# %% logging surpress
transformers.logging.set_verbosity(transformers.logging.ERROR)
logging.disable(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


# %% definitions
base_model = "roberta-base"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# %% prepare model & data
babe = load_dataset("mediabiasgroup/BABE")["train"].to_pandas()
pool = load_dataset("mediabiasgroup/BABE-icl-pool")["train"].to_pandas()
babe = (
    babe.merge(
        pool["text"],
        on="text",
        how="left",
        indicator=True,
    )
    .query(
        '_merge == "left_only"',
    )
    .drop("_merge", axis=1)
)

coreset_small = load_dataset("mandlc/anno-lexical-coreset-small-cluster")
coreset_big = load_dataset("mandlc/anno-lexical-coreset-big-cluster")

coreset_small_train = load_dataset("mandlc/anno-lexical-train-coreset-small-cluster")
coreset_big_train = load_dataset("mandlc/anno-lexical-train-coreset-big-cluster")

anno_lex = load_dataset("mediabiasgroup/anno-lexical")


model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=2,
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# %% tokenize data
tok = tokenizer(
    list(babe["text"]),
    truncation=True,
    padding=True,
    max_length=128,
)
babe_t = pd.DataFrame(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": babe["label"],
    },
)

tok = tokenizer(
    list(anno_lex["train"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
)
anno_lex_train_t = pd.DataFrame(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": anno_lex["train"]["label"],
    },
)

tok = tokenizer(
    list(coreset_small_train["train"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
)
coreset_small_train_set_t = pd.DataFrame(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": coreset_small_train["train"]["label"],
    },
)

tok = tokenizer(
    list(coreset_big_train["train"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
)
coreset_big_train_set_t = pd.DataFrame(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": coreset_big_train["train"]["label"],
    },
)


tok = tokenizer(
    list(coreset_small["train"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
)
coreset_small_train_t = pd.DataFrame(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": coreset_small["train"]["label"],
    },
)


tok = tokenizer(
    list(coreset_big["train"]["text"]),
    truncation=True,
    padding=True,
    max_length=128,
)
coreset_big_train_t = pd.DataFrame(
    {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "label": coreset_big["train"]["label"],
    },
)


# %% Training
training_args = TrainingArguments(
    output_dir="../data/training/training_cache/",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    logging_steps=50,
    disable_tqdm=False,
    save_total_limit=0,
    weight_decay=0.1,
    warmup_ratio=0.1,
    learning_rate=4e-5,
)


# %% define cv
@wandb_run()
def run_cv(run_name):
    scores = []
    for train_index, val_index in skfold.split(
        babe_t["input_ids"],
        babe["label"],
    ):
        if run_name == "baseline":
            train_t = Dataset.from_dict(babe_t.iloc[train_index])
        elif run_name == "coreset-small":
            train_t = Dataset.from_dict(coreset_small_train_t.iloc[train_index])
        elif run_name == "coreset-big":
            train_t = Dataset.from_dict(coreset_big_train_t.iloc[train_index])
        elif run_name == "coreset-small-train":
            train_t = Dataset.from_dict(coreset_small_train_set_t.iloc[train_index])
        elif run_name == "coreset-big-train":
            train_t = Dataset.from_dict(coreset_big_train_set_t.iloc[train_index])
        elif run_name == "anno-lexical":
            train_t = Dataset.from_pandas(anno_lex_train_t)
        else:
            raise NotImplementedError(f"unkown run_name: {run_name}")

        dev_t = Dataset.from_dict(babe_t.iloc[val_index])
        set_random_seed()
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=2,
        )
        model.to(device)
        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_t,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        trainer.train()

        # evaluation
        eval_dataloader = DataLoader(
            dev_t,
            batch_size=32,
            collate_fn=data_collator,
        )
        scores.append(compute_metrics(eval_dataloader, model))
        wandb.log(scores[-1])

    df = pd.DataFrame(scores)
    final_values = df.mean().to_dict()
    wandb.log({"final_" + k: v for k, v in final_values.items()})


# %%
run_cv(run_name="baseline")
run_cv(run_name="coreset-small")
run_cv(run_name="coreset-big")
run_cv(run_name="coreset-small-train")
run_cv(run_name="coreset-big-train")
run_cv(run_name="anno-lexical")
