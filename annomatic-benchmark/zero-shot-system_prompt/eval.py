#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import re
from typing import List, Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def is_contained(inner_start, inner_end, outer_start, outer_end):
    return (
        outer_start <= inner_start <= outer_end
        and outer_start <= inner_end <= outer_end
        and (inner_start, inner_end) != (outer_start, outer_end)
    )


def find_all_occurrences_indices(
    sentence: str,
    label: str,
) -> List[Tuple[int, int]]:
    """
    Finds all occurrences of a label in a sentence and returns a list of tuples
    with the start and end indices of the occurrences.

    Args:
        sentence: The sentence to search in.
        label: The label to search for.

    Returns:
        A list of tuples with the start and end indices of the occurrences.
    """
    return [
        (int(match.start()), int(match.end()))
        for match in re.finditer(label, sentence)
    ]


def find_contained_labels(labels):
    """
    Finds all labels that are contained in other labels.

    Args:
        labels: A list of labels.

    Returns:
        A dictionary with the labels as keys and a list of labels that are
        contained in the key label as values.
    """
    return {
        label_1: [
            label2
            for label2 in labels
            if label_1 in label2 and label_1 != label2
        ]
        for label_1 in labels
    }


def find_labels_in_sentence(
    sentence: str,
    labels: List[str],
) -> List[List[Tuple[int, int]]]:
    """
    Finds all occurrences of the labels in a sentence and returns a list of
    lists with the occurrences of each label.

    Labels that are contained in other labels are not returned.

    Args:
        sentence: The sentence to search in.
        labels: The labels to search for.

    Returns:
        A list of lists with the positional occurrence of each label.
    """
    if not labels:
        return []

    containable_map = find_contained_labels(labels)
    sentence_lower = sentence.lower()
    occurrences = [
        find_all_occurrences_indices(sentence_lower, label.lower())
        for label in labels
    ]
    label_occurrences = dict(zip(labels, occurrences))
    included_list = []
    for label, inner_pos in zip(labels, occurrences):
        contained = set()
        containing_labels = containable_map.get(label, set())

        for bigger_label in containing_labels:
            locations: List[Tuple[int, int]] = label_occurrences.get(
                bigger_label,
                list(),
            )

            for inner_start, inner_end in inner_pos:
                for outer_start, outer_end in locations:
                    if is_contained(
                        inner_start,
                        inner_end,
                        outer_start,
                        outer_end,
                    ):
                        contained.add((inner_start, inner_end))
        included_list.append(list(contained))

    return [
        list(set(pos) - set(included))
        for pos, included in zip(occurrences, included_list)
    ]


def find_label(
    sentence: str,
    labels: List[str],
    default_label: str = "?",
) -> str:
    """
    Search for given labels in the sentence and returns it if found. If only
    one label occur in the sentence, it will be returned. If no label or
    different labels occur in the sentence, '?' is returned.

    Args:
        sentence: The sentence to search in.
        labels: The labels to search for.
        default_label: The label to return if no label or different labels
            occur in the sentence.

    Returns:
        The label that occurs in the sentence or '?' if no label occurs in the
        sentence.
    """
    occurrences = find_labels_in_sentence(sentence=sentence, labels=labels)
    non_empty_indices = [i for i, sublist in enumerate(occurrences) if sublist]
    return (
        labels[non_empty_indices[0]]
        if len(
            non_empty_indices,
        )
        == 1
        else default_label
    )


def _soft_parse(
    df: pd.DataFrame,
    in_col: str,
    parsed_col: str,
    labels: List[str] = None,
) -> pd.DataFrame:
    if labels is None:
        raise ValueError("Labels are not set!")

    df[parsed_col] = df[in_col].apply(
        lambda x: find_label(x, labels),
    )


# In[2]:


# load

# load results
df_falcon_7b = pd.read_csv(f"./data/falcon-7b-instruct.csv")
df_flan_ul2 = pd.read_csv(f"./data/flan-ul2.csv")
df_flan_t5_base = pd.read_csv("./data/flan-t5-base.csv")
df_flan_t5_large = pd.read_csv("./data/flan-t5-large.csv")
df_flan_t5_xl = pd.read_csv("./data/flan-t5-xl.csv")
df_openai_gpt_3_5_turbo = pd.read_csv("./data/gpt-3.5-turbo.csv")
df_openai_gpt_4_turbo = pd.read_csv(f"./data/gpt-4.csv")
df_Llama_2_7b = pd.read_csv(f"./data/Llama-2-7b-chat-hf.csv")
df_Llama_2_13b = pd.read_csv(f"./data/Llama-2-13b-chat-hf.csv")
df_mistral_7b = pd.read_csv(f"./data/Mistral-7B-Instruct-v0.1.csv")
df_mixtral_8x7b = pd.read_csv(f"./data/Mixtral-8x7B-Instruct-v0.1.csv")
df_openchat_3_5 = pd.read_csv(f"./data/openchat_3.5.csv")
df_zephyr_7b_beta = pd.read_csv(f"./data/zephyr-7b-beta.csv")

# load pool
pool = load_dataset("mediabiasgroup/BABE-icl-pool")["train"].to_pandas()

# exclude pool from model (if needed)
df_falcon_7b = (
    df_falcon_7b.merge(
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
df_flan_ul2 = (
    df_flan_ul2.merge(
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
df_openai_gpt_3_5_turbo = (
    df_openai_gpt_3_5_turbo.merge(
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
df_openai_gpt_4_turbo = (
    df_openai_gpt_4_turbo.merge(
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
df_Llama_2_7b = (
    df_Llama_2_7b.merge(
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
df_Llama_2_13b = (
    df_Llama_2_13b.merge(
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
df_mistral_7b = (
    df_mistral_7b.merge(
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
df_mixtral_8x7b = (
    df_mixtral_8x7b.merge(
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
df_openchat_3_5 = (
    df_openchat_3_5.merge(
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
df_zephyr_7b_beta = (
    df_zephyr_7b_beta.merge(
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


# load babe
dataset = load_dataset("mediabiasgroup/BABE")
df_babe = pd.DataFrame(dataset["train"])

# df_merge_all_runs = only contains the elements legal in all annotations
df_merge_all_runs = df_babe

# df_merge_all_runs_with_errors = only contains the elements legal in all annotations
df_merge_all_runs_with_errors = df_babe


def find_first_occurrence(
    string,
    instruction="Instruction:",
    statement1="The answer is BIASED",
    statement2="The answer is NOT BIASED",
):
    index_instruction = string.find(instruction)
    index1 = string.find(statement1)
    index2 = string.find(statement2)

    if index1 == -1 and index2 == -1:
        return "Neither instruction nor statements found in the given string."
    elif index1 == -1 and index2 == -1:
        return "Neither statement found in the given string."
    elif index1 == -1:
        if index_instruction == -1 or index2 < index_instruction:
            return f"{statement2}"
        else:
            return "After Instruction"
    elif index2 == -1:
        if index_instruction == -1 or index1 < index_instruction:
            return f"{statement1}"
        else:
            return "After Instruction"
    elif index1 < index2:
        if index_instruction == -1 or index1 < index_instruction:
            return f"{statement1}"
        else:
            return "After Instruction"
    else:
        if index_instruction == -1 or index2 < index_instruction:
            return f"{statement2}"
        else:
            return "After Instruction"


# # Falcon 7B

# In[3]:


_soft_parse(
    df_falcon_7b,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_falcon_7b.query("label == '?'")


# In[4]:


# preprocessing
def update_label(row):
    if row["response"].startswith("'BIASED'") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("'Biased'") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("1") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("'NOT BIASED'") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].startswith("'Not BIASED'") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].startswith("'Not Biased'") and row["label"] == "?":
        return "NOT BIASED"

    else:
        return row["label"]


df_falcon_7b["label"] = df_falcon_7b.apply(update_label, axis=1)
df_falcon_7b.query("label == '?'")


# In[5]:


# preprocessing

df_falcon_7b["label"] = df_falcon_7b.apply(update_label, axis=1)
df_falcon_7b.query("label == '?'")


# In[6]:


df_falcon_7b = df_falcon_7b.rename(columns={"label": "falcon_7b_label"})
df_falcon_7b["falcon_7b_label"] = df_falcon_7b["falcon_7b_label"].replace(
    "BIASED", 1
)
df_falcon_7b["falcon_7b_label"] = df_falcon_7b["falcon_7b_label"].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_falcon_7b[df_falcon_7b["falcon_7b_label"] != "?"][
        ["text", "falcon_7b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_falcon_7b[df_falcon_7b["falcon_7b_label"] != "?"][
        ["text", "falcon_7b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_falcon_7b[["text", "falcon_7b_label"]], on="text"
)

ground_truth = df_merge["label"].astype(int)
falcon_7b_label = df_merge["falcon_7b_label"].astype(int)


# In[7]:


print(
    "F1-Score with Falcon 7b with (0 shot + Sys. Prompt): ",
    f1_score(ground_truth, falcon_7b_label),
)
print(
    "Precision with Falcon 7b with (0 shot + Sys. Prompt): ",
    precision_score(ground_truth, falcon_7b_label),
)
print(
    "Recall with Falcon 7b with (0 shot + Sys. Prompt): ",
    recall_score(ground_truth, falcon_7b_label),
)
print(
    "Accuracy with Falcon 7b with (0 shot + Sys. Prompt): ",
    accuracy_score(ground_truth, falcon_7b_label),
)


# # Flan UL2

# In[8]:


_soft_parse(
    df_flan_ul2,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_flan_ul2.query("label == '?'")


# In[9]:


df_flan_ul2 = df_flan_ul2.rename(columns={"label": "flan_ul2_label"})
df_flan_ul2["flan_ul2_label"] = df_flan_ul2["flan_ul2_label"].replace(
    "BIASED", 1
)
df_flan_ul2["flan_ul2_label"] = df_flan_ul2["flan_ul2_label"].replace(
    "NOT BIASED", 0
)

df_merge = df_babe.merge(
    df_flan_ul2[df_flan_ul2["flan_ul2_label"] != "?"][
        ["text", "flan_ul2_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_flan_ul2[df_flan_ul2["flan_ul2_label"] != "?"][
        ["text", "flan_ul2_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_flan_ul2[["text", "flan_ul2_label"]], on="text"
)

ground_truth = df_merge["label"].astype(int)
flan_ul2_label = df_merge["flan_ul2_label"].astype(int)


# In[10]:


print(
    "F1-Score with Flan UL2 (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, flan_ul2_label),
)
print(
    "Precision with Flan UL2 (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, flan_ul2_label),
)
print(
    "Recall with Flan UL2 (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, flan_ul2_label),
)
print(
    "Accuracy with Flan UL2 (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, flan_ul2_label),
)


# # Flan T5 base

# In[11]:


_soft_parse(
    df_flan_t5_base,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_flan_t5_base.query("label == '?'")


# In[12]:


df_flan_t5_base = df_flan_t5_base.rename(
    columns={"label": "flan_t5_base_label"}
)
df_flan_t5_base["flan_t5_base_label"] = df_flan_t5_base[
    "flan_t5_base_label"
].replace("BIASED", 1)
df_flan_t5_base["flan_t5_base_label"] = df_flan_t5_base[
    "flan_t5_base_label"
].replace("NOT BIASED", 0)

df_merge = df_babe.merge(
    df_flan_t5_base[df_flan_t5_base["flan_t5_base_label"] != "?"][
        ["text", "flan_t5_base_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_flan_t5_base[df_flan_t5_base["flan_t5_base_label"] != "?"][
        ["text", "flan_t5_base_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_flan_t5_base[["text", "flan_t5_base_label"]], on="text"
)

ground_truth = df_merge["label"].astype(int)
flan_t5_base_label = df_merge["flan_t5_base_label"].astype(int)


# In[13]:


print(
    "F1-Score with Flan T5 base (0 shot): ",
    f1_score(ground_truth, flan_t5_base_label),
)
print(
    "Precision with Flan T5 base (0 shot): ",
    precision_score(ground_truth, flan_t5_base_label),
)
print(
    "Recall with Flan T5 base (0 shot): ",
    recall_score(ground_truth, flan_t5_base_label),
)
print(
    "Accuracy with Flan T5 base (0 shot): ",
    accuracy_score(ground_truth, flan_t5_base_label),
)


# # Flan T5 large

# In[14]:


_soft_parse(
    df_flan_t5_large,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_flan_t5_large.query("label == '?'")


# In[15]:


df_flan_t5_large = df_flan_t5_large.rename(
    columns={"label": "flan_t5_large_label"}
)
df_flan_t5_large["flan_t5_large_label"] = df_flan_t5_large[
    "flan_t5_large_label"
].replace("BIASED", 1)
df_flan_t5_large["flan_t5_large_label"] = df_flan_t5_large[
    "flan_t5_large_label"
].replace("NOT BIASED", 0)

df_merge = df_babe.merge(
    df_flan_t5_large[df_flan_t5_large["flan_t5_large_label"] != "?"][
        ["text", "flan_t5_large_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_flan_t5_large[df_flan_t5_large["flan_t5_large_label"] != "?"][
        ["text", "flan_t5_large_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_flan_t5_large[["text", "flan_t5_large_label"]], on="text"
)

ground_truth = df_merge["label"].astype(int)
flan_t5_large_label = df_merge["flan_t5_large_label"].astype(int)


# In[16]:


print(
    "F1-Score with Flan T5 Large (0 shot + Sys Prompt): ",
    f1_score(ground_truth, flan_t5_large_label),
)
print(
    "Precision with Flan T5 Large (0 shot + Sys Prompt): ",
    precision_score(ground_truth, flan_t5_large_label),
)
print(
    "Recall with Flan T5 Large (0 shot + Sys Prompt): ",
    recall_score(ground_truth, flan_t5_large_label),
)
print(
    "Accuracy with Flan T5 Large (0 shot + Sys Prompt): ",
    accuracy_score(ground_truth, flan_t5_large_label),
)


# # Flan T5 Xl

# In[17]:


_soft_parse(
    df_flan_t5_xl,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_flan_t5_xl.query("label == '?'")


# In[18]:


df_flan_t5_xl.loc[[1705], "label"] = "NOT BIASED"


# In[19]:


df_flan_t5_xl = df_flan_t5_xl.rename(columns={"label": "flan_t5_xl_label"})
df_flan_t5_xl["flan_t5_xl_label"] = df_flan_t5_xl["flan_t5_xl_label"].replace(
    "BIASED", 1
)
df_flan_t5_xl["flan_t5_xl_label"] = df_flan_t5_xl["flan_t5_xl_label"].replace(
    "NOT BIASED", 0
)

df_merge = df_babe.merge(
    df_flan_t5_xl[df_flan_t5_xl["flan_t5_xl_label"] != "?"][
        ["text", "flan_t5_xl_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_flan_t5_xl[df_flan_t5_xl["flan_t5_xl_label"] != "?"][
        ["text", "flan_t5_xl_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_flan_t5_xl[["text", "flan_t5_xl_label"]], on="text"
)

ground_truth = df_merge["label"].astype(int)
flan_t5_xl_label = df_merge["flan_t5_xl_label"].astype(int)


# In[20]:


print(
    "F1-Score with Flan T5 xl (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, flan_t5_xl_label),
)
print(
    "Precision with Flan T5 xl (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, flan_t5_xl_label),
)
print(
    "Recall with Flan T5 xl (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, flan_t5_xl_label),
)
print(
    "Accuracy with Flan T5 xl (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, flan_t5_xl_label),
)


# # GPT-3.5-turbo

# In[21]:


_soft_parse(
    df_openai_gpt_3_5_turbo,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_openai_gpt_3_5_turbo.query("label == '?'")


# In[22]:


df_openai_gpt_3_5_turbo = df_openai_gpt_3_5_turbo.rename(
    columns={"label": "gpt_3_5_label"}
)
df_openai_gpt_3_5_turbo["gpt_3_5_label"] = df_openai_gpt_3_5_turbo[
    "gpt_3_5_label"
].replace(
    "BIASED",
    1,
)
df_openai_gpt_3_5_turbo["gpt_3_5_label"] = df_openai_gpt_3_5_turbo[
    "gpt_3_5_label"
].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_openai_gpt_3_5_turbo[df_openai_gpt_3_5_turbo["gpt_3_5_label"] != "?"][
        ["text", "gpt_3_5_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_openai_gpt_3_5_turbo[df_openai_gpt_3_5_turbo["gpt_3_5_label"] != "?"][
        ["text", "gpt_3_5_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_openai_gpt_3_5_turbo[["text", "gpt_3_5_label"]], on="text"
)


ground_truth = df_merge["label"].astype(int)
gpt_3_5_label = df_merge["gpt_3_5_label"].astype(int)


# In[23]:


print(
    "F1-Score with GPT 3.5 Turbo with (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, gpt_3_5_label),
)
print(
    "Precision with GPT 3.5 Turbo with (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, gpt_3_5_label),
)
print(
    "Recall with GPT 3.5 Turbo with (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, gpt_3_5_label),
)
print(
    "Accuracy with GPT 3.5 Turbo with (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, gpt_3_5_label),
)


# # GPT 4 - turbo

# In[24]:


_soft_parse(
    df_openai_gpt_4_turbo,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_openai_gpt_4_turbo.query("label == '?'")


# In[25]:


df_openai_gpt_4_turbo = df_openai_gpt_4_turbo.rename(
    columns={"label": "gpt_4_label"}
)
df_openai_gpt_4_turbo["gpt_4_label"] = df_openai_gpt_4_turbo[
    "gpt_4_label"
].replace(
    "BIASED",
    1,
)
df_openai_gpt_4_turbo["gpt_4_label"] = df_openai_gpt_4_turbo[
    "gpt_4_label"
].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_openai_gpt_4_turbo[df_openai_gpt_4_turbo["gpt_4_label"] != "?"][
        ["text", "gpt_4_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_openai_gpt_4_turbo[df_openai_gpt_4_turbo["gpt_4_label"] != "?"][
        ["text", "gpt_4_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_openai_gpt_4_turbo[["text", "gpt_4_label"]], on="text"
)


ground_truth = df_merge["label"].astype(int)
gpt_4_label = df_merge["gpt_4_label"].astype(int)


# In[26]:


print(
    "F1-Score with GPT 4 turbo with (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, gpt_4_label),
)
print(
    "Precision with GPT 4 turbo with (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, gpt_4_label),
)
print(
    "Recall with GPT 4 turbo with (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, gpt_4_label),
)
print(
    "Accuracy with GPT 4 turbo with (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, gpt_4_label),
)


# # Llama-2-7b-chat-hf

# In[27]:


_soft_parse(
    df_Llama_2_7b,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_Llama_2_7b.query("label == '?'")


# In[28]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_Llama_2_7b["label"] = df_Llama_2_7b.apply(update_label, axis=1)
df_Llama_2_7b.query("label == '?'")


# In[29]:


df_Llama_2_7b = df_Llama_2_7b.rename(columns={"label": "llama_7b_label"})
df_Llama_2_7b["llama_7b_label"] = df_Llama_2_7b["llama_7b_label"].replace(
    "BIASED", 1
)
df_Llama_2_7b["llama_7b_label"] = df_Llama_2_7b["llama_7b_label"].replace(
    "NOT BIASED", 0
)

df_merge = df_babe.merge(
    df_Llama_2_7b[df_Llama_2_7b["llama_7b_label"] != "?"][
        ["text", "llama_7b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_Llama_2_7b[df_Llama_2_7b["llama_7b_label"] != "?"][
        ["text", "llama_7b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_Llama_2_7b[["text", "llama_7b_label"]], on="text"
)


ground_truth = df_merge["label"].astype(int)
llama_7b_label = df_merge["llama_7b_label"].astype(int)


# In[30]:


print(
    "F1-Score with llama 7b (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, llama_7b_label),
)
print(
    "Precision with llama 7b (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, llama_7b_label),
)
print(
    "Recall with llama 7b (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, llama_7b_label),
)
print(
    "Accuracy with llama 7b (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, llama_7b_label),
)


# # Llama-2-13b-chat-hf

# In[31]:


_soft_parse(
    df_Llama_2_13b,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_Llama_2_13b.query("label == '?'")


# In[32]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "Please explain your answer.".lower() in row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Please provide your answer.".lower() in row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Please provide a brief explanation for your answer.".lower()
        in row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Please provide a brief explanation of your answer.".lower()
        in row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Please provide your answer in the format".lower()
        in row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Explain your answer".lower() in row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Explain your reasoning".lower() in row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Thank you!".lower() in row["response"].lower() and row["label"] == "?"
    ):
        return "!"
    elif (
        "1 for BIASED or 0 for NOT BIASED.".lower() == row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "1 for BIASED, 0 for NOT BIASED.".lower() == row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    else:
        return row["label"]


df_Llama_2_13b["label"] = df_Llama_2_13b.apply(update_label, axis=1)
df_Llama_2_13b.query("label == '?'")


# In[33]:


# map "!" flag back to "?"

df_Llama_2_13b.loc[[2974], "label"] = "BIASED"
df_Llama_2_13b.loc[[], "label"] = "NOT BIASED"

df_Llama_2_13b.loc[
    [
        3739,
        3405,
        3136,
        3132,
        2854,
        2662,
        2165,
        1943,
        1930,
        1775,
        1717,
        962,
        897,
        811,
        782,
        498,
        175,
    ],
    "label",
] = "!"

# undefined
df_Llama_2_13b.loc[498]["response"]
df_Llama_2_13b.query("label == '?'")


# In[34]:


# map "!" flag back to "?"
df_Llama_2_13b["label"] = df_Llama_2_13b["label"].replace("!", "?")

df_Llama_2_13b = df_Llama_2_13b.rename(columns={"label": "llama_13b_label"})
df_Llama_2_13b["llama_13b_label"] = df_Llama_2_13b["llama_13b_label"].replace(
    "BIASED", 1
)
df_Llama_2_13b["llama_13b_label"] = df_Llama_2_13b["llama_13b_label"].replace(
    "NOT BIASED", 0
)

df_merge = df_babe.merge(
    df_Llama_2_13b[df_Llama_2_13b["llama_13b_label"] != "?"][
        ["text", "llama_13b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_Llama_2_13b[df_Llama_2_13b["llama_13b_label"] != "?"][
        ["text", "llama_13b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_Llama_2_13b[["text", "llama_13b_label"]], on="text"
)


ground_truth = df_merge["label"].astype(int)
llama_13b_label = df_merge["llama_13b_label"].astype(int)


# In[35]:


print(
    "F1-Score with TODO with (0 shot + Sys Prompt): ",
    f1_score(ground_truth, llama_13b_label),
)
print(
    "Precision with TODO with (0 shot + Sys Prompt): ",
    precision_score(ground_truth, llama_13b_label),
)
print(
    "Recall with TODO with (0 shot + Sys Prompt): ",
    recall_score(ground_truth, llama_13b_label),
)
print(
    "Accuracy with TODO with (0 shot + Sys Prompt): ",
    accuracy_score(ground_truth, llama_13b_label),
)


# # Mistral-7B-Instruct-v0.1

# In[36]:


_soft_parse(
    df_mistral_7b,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_mistral_7b.query("label == '?'")


# In[37]:


df_mistral_7b = df_mistral_7b.rename(columns={"label": "mistral_7b_label"})
df_mistral_7b["mistral_7b_label"] = df_mistral_7b["mistral_7b_label"].replace(
    "BIASED", 1
)
df_mistral_7b["mistral_7b_label"] = df_mistral_7b["mistral_7b_label"].replace(
    "NOT BIASED", 0
)

df_merge = df_babe.merge(
    df_mistral_7b[df_mistral_7b["mistral_7b_label"] != "?"][
        ["text", "mistral_7b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_mistral_7b[df_mistral_7b["mistral_7b_label"] != "?"][
        ["text", "mistral_7b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_mistral_7b[["text", "mistral_7b_label"]], on="text"
)


ground_truth = df_merge["label"].astype(int)
df_mistral_7b_label = df_merge["mistral_7b_label"].astype(int)


# In[38]:


print(
    "F1-Score with Mistral-7B-Instruct-v0.1 with (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, df_mistral_7b_label),
)
print(
    "Precision with Mistral-7B-Instruct-v0.1 with (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, df_mistral_7b_label),
)
print(
    "Recall with Mistral-7B-Instruct-v0.1 with (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, df_mistral_7b_label),
)
print(
    "Accuracy with Mistral-7B-Instruct-v0.1 with (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, df_mistral_7b_label),
)


# # Mixtral-8x7B

# In[39]:


df_mixtral_8x7b.query("label == '?'")


# In[40]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    else:
        return row["label"]


df_mixtral_8x7b["label"] = df_mixtral_8x7b.apply(update_label, axis=1)
df_mixtral_8x7b.query("label == '?'")


# In[41]:


df_mixtral_8x7b.loc[
    [583, 1916, 1960, 1984, 2316, 2783, 2825, 2938, 3892, 2035], "label"
] = "BIASED"
df_mixtral_8x7b.loc[[239, 2287], "label"] = "NOT BIASED"

# undefined

# df_mixtral_8x7b.loc[3892]['response']
df_mixtral_8x7b.query("label == '?'")


# In[42]:


df_mixtral_8x7b = df_mixtral_8x7b.rename(
    columns={"label": "mixtral_8x7b_label"}
)
df_mixtral_8x7b["mixtral_8x7b_label"] = df_mixtral_8x7b[
    "mixtral_8x7b_label"
].replace("BIASED", 1)
df_mixtral_8x7b["mixtral_8x7b_label"] = df_mixtral_8x7b[
    "mixtral_8x7b_label"
].replace(
    "NOT BIASED",
    0,
)

df_merge = df_babe.merge(
    df_mixtral_8x7b[df_mixtral_8x7b["mixtral_8x7b_label"] != "?"][
        ["text", "mixtral_8x7b_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_mixtral_8x7b[df_mixtral_8x7b["mixtral_8x7b_label"] != "?"][
        ["text", "mixtral_8x7b_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_mixtral_8x7b[["text", "mixtral_8x7b_label"]], on="text"
)


ground_truth = df_merge["label"].astype(int)
df_mixtral_8x7b_label = df_merge["mixtral_8x7b_label"].astype(int)


# In[43]:


print(
    "F1-Score with mixtral_8x7b with (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Precision with mixtral_8x7b with (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Recall with mixtral_8x7b with (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Accuracy with mixtral_8x7b with (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, df_mixtral_8x7b_label),
)


# # OpenChat_3.5

# In[44]:


_soft_parse(
    df_openchat_3_5,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_openchat_3_5.query("label == '?'")


# In[45]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    if row["response"].startswith("BIAS") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    else:
        return row["label"]


df_openchat_3_5["label"] = df_openchat_3_5.apply(update_label, axis=1)
df_openchat_3_5.query("label == '?'")


# In[46]:


df_openchat_3_5 = df_openchat_3_5.rename(columns={"label": "openchat_label"})
df_openchat_3_5["openchat_label"] = df_openchat_3_5["openchat_label"].replace(
    "BIASED", 1
)
df_openchat_3_5["openchat_label"] = df_openchat_3_5["openchat_label"].replace(
    "NOT BIASED", 0
)

df_merge = df_babe.merge(
    df_openchat_3_5[df_openchat_3_5["openchat_label"] != "?"][
        ["text", "openchat_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_openchat_3_5[df_openchat_3_5["openchat_label"] != "?"][
        ["text", "openchat_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_openchat_3_5[["text", "openchat_label"]], on="text"
)


ground_truth = df_merge["label"].astype(int)
openchat_label = df_merge["openchat_label"].astype(int)


# In[47]:


print(
    "F1-Score with OpenChat 3.5 with (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, openchat_label),
)
print(
    "Precision with OpenChat 3.5 with (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, openchat_label),
)
print(
    "Recall with OpenChat 3.5 with (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, openchat_label),
)
print(
    "Accuracy with OpenChat 3.5 with (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, openchat_label),
)


# # zephyr-7b-beta

# In[48]:


_soft_parse(
    df_zephyr_7b_beta,
    "response",
    "label",
    ["BIASED", "NOT BIASED"],
)
df_zephyr_7b_beta.query("label == '?'")


# In[49]:


# preprocessing
def update_label(row):
    if row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence above is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence above is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence above is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif row["response"].startswith("100% BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("100% NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED"
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED"
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence above is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].startswith("100% BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].startswith(
        f"The sentence \"{row['text']}\" is NOT BIASED"
    ):
        return "NOT BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_zephyr_7b_beta["label"] = df_zephyr_7b_beta.apply(update_label, axis=1)
df_zephyr_7b_beta.query("label == '?'")


# In[50]:


df_zephyr_7b_beta.loc[
    [
        151,
        206,
        516,
        842,
        1111,
        1422,
        1684,
        1701,
        1712,
        1961,
        2143,
        2434,
        2470,
        3435,
        3512,
    ],
    "label",
] = "NOT BIASED"
df_zephyr_7b_beta.loc[[2981], "label"] = "BIASED"
df_zephyr_7b_beta.query("label == '?'")


# In[51]:


df_zephyr_7b_beta = df_zephyr_7b_beta.rename(columns={"label": "zephyr_label"})
df_zephyr_7b_beta["zephyr_label"] = df_zephyr_7b_beta["zephyr_label"].replace(
    "BIASED", 1
)
df_zephyr_7b_beta["zephyr_label"] = df_zephyr_7b_beta["zephyr_label"].replace(
    "NOT BIASED", 0
)

df_merge = df_babe.merge(
    df_zephyr_7b_beta[df_zephyr_7b_beta["zephyr_label"] != "?"][
        ["text", "zephyr_label"]
    ],
    on="text",
)
df_merge_all_runs = df_merge_all_runs.merge(
    df_zephyr_7b_beta[df_zephyr_7b_beta["zephyr_label"] != "?"][
        ["text", "zephyr_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_zephyr_7b_beta[["text", "zephyr_label"]], on="text"
)

ground_truth = df_merge["label"].astype(int)
zephyr_label = df_merge["zephyr_label"].astype(int)


# In[52]:


print(
    "F1-Score with zephyr beta (0 Shot + Sys Prompt): ",
    f1_score(ground_truth, zephyr_label),
)
print(
    "Precision with zephyr beta (0 Shot + Sys Prompt): ",
    precision_score(ground_truth, zephyr_label),
)
print(
    "Recall with zephyr beta (0 Shot + Sys Prompt): ",
    recall_score(ground_truth, zephyr_label),
)
print(
    "Accuracy with zephyr beta (0 Shot + Sys Prompt): ",
    accuracy_score(ground_truth, zephyr_label),
)


# In[53]:


# safe the file
df_merge_all_runs_with_errors.to_csv("./all_runs_with_errors.csv", index=False)
