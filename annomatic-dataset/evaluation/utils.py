import functools
import random

import numpy as np
import torch
import wandb
from config import PROJECT_NAME, RANDOM_SEED
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def set_random_seed(seed=RANDOM_SEED):
    """Random seed for comparable results."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def compute_metrics(testing_dataloader, model):
    """computes F1 score and accuracy over dataset

    Args:
        model (any type): model for evaluation
        testing_dataloader (huggingface dataset): self explained

    Returns:
        dict
    """

    y_true = []
    y_pred = []

    model.eval()
    for batch in testing_dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_true.extend(batch["labels"].tolist())
        y_pred.extend(predictions.tolist())

    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return {
        "mcc": mcc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


def compute_metrics_hf(preds):
    """computes F1 score and accuracy over dataset

    Args:
        model (any type): model for evaluation
        testing_dataloader (huggingface dataset): self explained

    Returns:
        dict
    """

    y_pred = preds.predictions
    y_true = preds.label_ids

    y_pred = y_pred.argmax(axis=1)

    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return {
        "mcc": mcc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


def wandb_run():
    def decorator_wandb_run(func):
        @functools.wraps(func)
        def wrapper(run_name, *args, **kwargs):
            wandb.init(
                entity="media-bias-group", project=PROJECT_NAME, name=run_name
            )
            func(run_name, *args, **kwargs)
            wandb.finish()

        return wrapper

    return decorator_wandb_run
