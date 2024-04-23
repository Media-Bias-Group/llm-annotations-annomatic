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


def split_and_store(row):
    split_parts = row["response"].split("\n\nInstruction", 1)
    return split_parts[0]


# In[2]:


# load

# load results
df_falcon_7b = pd.read_csv(f"./data/falcon-7b-instruct.csv")
df_flan_t5_base = pd.read_csv("./data/flan-t5-base.csv")
df_flan_t5_large = pd.read_csv("./data/flan-t5-large.csv")
df_flan_t5_xl = pd.read_csv("./data/flan-t5-xl.csv")
df_flan_ul2 = pd.read_csv(f"./data/flan-ul2.csv")
df_openai_gpt_3_5_turbo = pd.read_csv("./data/gpt-3.5-turbo.csv")
df_openai_gpt_4_turbo = pd.read_csv(f"./data/gpt-4-1106-preview.csv")
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

df_flan_t5_base = (
    df_flan_t5_base.merge(
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

df_flan_t5_large = (
    df_flan_t5_large.merge(
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

df_flan_t5_xl = (
    df_flan_t5_xl.merge(
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


df_falcon_7b.query("label == '?'")


# In[4]:


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
        row["response"].startswith("The statement is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is labeled as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence appears to be NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].endswith("is NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("is BIASED.") and row["label"] == "?":
        return "BIASED"
    elif (
        "can be classified as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "classify this sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify this sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "the sentence is NOT BIASED" in row["response"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "sentence is NOT biased" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "classified as SUBTLY BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif "appears to be NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("BIASED.") and row["label"] == "?":
        return "BIASED"
    elif row["response"].endswith("NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "NOT BIASED".lower() not in row["response"].lower()
        and "BIASED".lower() not in row["response"].lower()
    ) and row["label"] == "?":
        return "!"
    else:
        return row["label"]


df_falcon_7b["label"] = df_falcon_7b.apply(update_label, axis=1)
df_falcon_7b.query("label == '?'")


# In[5]:


# manual assignment
# if there is a tendency like 'may be biased/potential bias' or 'is factual' we classified it as biased or not biased respectively
df_falcon_7b.loc[
    [
        4008,
        3883,
        3829,
        3768,
        3725,
        3647,
        3611,
        3219,
        2904,
        2520,
        2034,
        2009,
        695,
        584,
        473,
    ],
    "label",
] = "BIASED"
df_falcon_7b.loc[
    [
        3960,
        3739,
        3670,
        3559,
        3508,
        3351,
        2876,
        2492,
        2098,
        1796,
        1786,
        1463,
        1325,
        1250,
        1063,
        1046,
        1040,
        890,
        829,
        622,
        351,
        298,
        108,
        56,
    ],
    "label",
] = "NOT BIASED"

df_falcon_7b.loc[
    [
        4053,
        3971,
        3920,
        3493,
        3319,
        3147,
        2730,
        2704,
        2561,
        2341,
        2240,
        2121,
        2102,
        1851,
        1846,
        1781,
        1550,
        1459,
        1441,
        1178,
        804,
        380,
        328,
        129,
        85,
    ],
    "label",
] = "!"


df_falcon_7b.query("label == '?'")

# Assessment (example)
# 1. Identify the subject of the sentence,
# 2a. Consider the source of the sentence
# 3. Analyze the sentence to see if it is presenting a balanced view of the situation. (or factual/ objective)
# 4. Look at the sentence's language and tone.
# 5. Consider the sentence's implications and whether it is likely to influence the reader's opinion.
# 6. Finally, determine whether the sentence is likely to be BIASED or NOT BIASED.


# In[ ]:


# In[6]:


# map "!" flag back to "?"
df_falcon_7b["label"] = df_falcon_7b["label"].replace("!", "?")

df_falcon_7b = df_falcon_7b.rename(columns={"label": "falcon_7b_label"})
df_falcon_7b["falcon_7b_label"] = df_falcon_7b["falcon_7b_label"].replace(
    "BIASED",
    1,
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
    df_falcon_7b[["text", "falcon_7b_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
falcon_7b_label = df_merge["falcon_7b_label"].astype(int)


# In[7]:


print(
    "F1-Score with Falcon 7b with (0 shot CoT): ",
    f1_score(ground_truth, falcon_7b_label),
)
print(
    "Precision with Falcon 7b with (0 shot CoT): ",
    precision_score(ground_truth, falcon_7b_label),
)
print(
    "Recall with Falcon 7b with (0 shot CoT): ",
    recall_score(ground_truth, falcon_7b_label),
)
print(
    "Accuracy with Falcon 7b with (0 shot CoT): ",
    accuracy_score(ground_truth, falcon_7b_label),
)


# # Flan T5 Base

# In[8]:


df_flan_t5_base.query("label == '?'")


# In[9]:


df_flan_t5_base = df_flan_t5_base.rename(
    columns={"label": "flan_t5_base_label"},
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
    df_flan_t5_base[["text", "flan_t5_base_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
flan_t5_base_label = df_merge["flan_t5_base_label"].astype(int)


# In[10]:


print(
    "F1-Score with Flan T5 base (0 shot CoT): ",
    f1_score(ground_truth, flan_t5_base_label),
)
print(
    "Precision with Flan T5 base (0 shot CoT): ",
    precision_score(ground_truth, flan_t5_base_label),
)
print(
    "Recall with Flan T5 base (0 shot CoT): ",
    recall_score(ground_truth, flan_t5_base_label),
)
print(
    "Accuracy with Flan T5 base (0 shot CoT): ",
    accuracy_score(ground_truth, flan_t5_base_label),
)


# # Flan T5 Large

# In[11]:


df_flan_t5_large.query("label == '?'")


# In[12]:


def update_label(row):
    if (
        str(row["response"]).startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).endswith("answer is BIASED.")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).endswith("answer is NOT BIASED.")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).endswith("The answer: BIASED.")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).endswith("The answer: NOT BIASED.")
        and row["label"] == "?"
    ):
        return "NOT BIASED"

    elif (
        "BIASED".lower() not in str(row["response"]).lower()
        and "NOT BIASED".lower() not in str(row["response"]).lower()
    ) and row["label"] == "?":
        return "!"
    elif (
        "BIASED is the opposite of not biased.".lower()
        == str(row["response"]).lower()
        and row["label"] == "?"
    ):
        return "!"
    else:
        return row["label"]


df_flan_t5_large["label"] = df_flan_t5_large.apply(update_label, axis=1)
df_flan_t5_large.query("label == '?'")


# In[13]:


df_flan_t5_large.response.loc[4009]


# In[14]:


# manual assignment
df_flan_t5_large.loc[[], "label"] = "BIASED"
df_flan_t5_large.loc[[3116], "label"] = "NOT BIASED"

df_flan_t5_large.loc[
    [
        110,
        166,
        296,
        1009,
        1341,
        1357,
        1400,
        1851,
        2378,
        1893,
        2402,
        3241,
        3300,
        3312,
        3339,
        4009,
    ],
    "label",
] = "!"


df_flan_t5_large.query("label == '?'")


# In[15]:


# map '!' flag back to "?"
df_flan_t5_large["label"] = df_flan_t5_large["label"].replace("!", "?")

df_flan_t5_large = df_flan_t5_large.rename(
    columns={"label": "flan_t5_large_label"},
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
    df_flan_t5_large[["text", "flan_t5_large_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
flan_t5_large_label = df_merge["flan_t5_large_label"].astype(int)


# In[16]:


print(
    "F1-Score with Flan T5 Large (0 shot CoT): ",
    f1_score(ground_truth, flan_t5_large_label),
)
print(
    "Precision with Flan T5 Large (0 shot CoT): ",
    precision_score(ground_truth, flan_t5_large_label),
)
print(
    "Recall with Flan T5 Large (0 shot CoT): ",
    recall_score(ground_truth, flan_t5_large_label),
)
print(
    "Accuracy with Flan T5 Large (0 shot CoT): ",
    accuracy_score(ground_truth, flan_t5_large_label),
)


# # Flan T5 XL

# In[17]:


df_flan_t5_xl.query("label == '?'")


# In[18]:


def update_label(row):
    if str(row["response"]).startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif str(row["response"]).startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif str(row["response"]).startswith("Not biased") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).endswith("answer is BIASED.")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).endswith("answer is NOT BIASED.")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_flan_t5_xl["label"] = df_flan_t5_xl.apply(update_label, axis=1)
df_flan_t5_xl.query("label == '?'")


# In[19]:


df_flan_t5_xl = df_flan_t5_xl.rename(columns={"label": "flan_t5_xl_label"})
df_flan_t5_xl["flan_t5_xl_label"] = df_flan_t5_xl["flan_t5_xl_label"].replace(
    "BIASED",
    1,
)
df_flan_t5_xl["flan_t5_xl_label"] = df_flan_t5_xl["flan_t5_xl_label"].replace(
    "NOT BIASED",
    0,
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
    df_flan_t5_xl[["text", "flan_t5_xl_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
flan_t5_xl_label = df_merge["flan_t5_xl_label"].astype(int)


# In[20]:


print(
    "F1-Score with Flan T5 xl (0 Shot CoT): ",
    f1_score(ground_truth, flan_t5_xl_label),
)
print(
    "Precision with Flan T5 xl (0 Shot CoT): ",
    precision_score(ground_truth, flan_t5_xl_label),
)
print(
    "Recall with Flan T5 xl (0 Shot CoT): ",
    recall_score(ground_truth, flan_t5_xl_label),
)
print(
    "Accuracy with Flan T5 xl (0 Shot CoT): ",
    accuracy_score(ground_truth, flan_t5_xl_label),
)


# # Flan UL2

# In[21]:


df_flan_ul2.query("label == '?'")


# In[22]:


def update_label(row):
    if str(row["response"]).startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif str(row["response"]).startswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif str(row["response"]).startswith("Not biased") and row["label"] == "?":
        return "NOT BIASED"
    elif str(row["response"]).startswith("Not Biased") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        str(row["response"]).startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        str(row["response"]).startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    else:
        return row["label"]


df_flan_ul2["label"] = df_flan_ul2.apply(update_label, axis=1)
df_flan_ul2.query("label == '?'")


# In[23]:


df_flan_ul2 = df_flan_ul2.rename(columns={"label": "flan_ul2_label"})
df_flan_ul2["flan_ul2_label"] = df_flan_ul2["flan_ul2_label"].replace(
    "BIASED",
    1,
)
df_flan_ul2["flan_ul2_label"] = df_flan_ul2["flan_ul2_label"].replace(
    "NOT BIASED",
    0,
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
    df_flan_ul2[["text", "flan_ul2_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
flan_ul2_label = df_merge["flan_ul2_label"].astype(int)


# In[24]:


print(
    "F1-Score with Flan UL2 (0 shot CoT): ",
    f1_score(ground_truth, flan_ul2_label),
)
print(
    "Precision with Flan UL2 (0 shot CoT): ",
    precision_score(ground_truth, flan_ul2_label),
)
print(
    "Recall with Flan UL2 (0 shot CoT): ",
    recall_score(ground_truth, flan_ul2_label),
)
print(
    "Accuracy with Flan UL2 (0 shot CoT): ",
    accuracy_score(ground_truth, flan_ul2_label),
)


# # GPT-3.5-turbo

# In[25]:


df_openai_gpt_3_5_turbo.query("label == '?'")


# In[26]:


def update_label(row):
    if (
        row["response"].startswith(
            "The sentence provided does not exhibit any explicit bias",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence provided does not contain any explicit bias",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence provided does not display any explicit bias.",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The given sentence does not exhibit any explicit bias.",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence above does not exhibit any explicit bias",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence provided does not explicitly display any bias",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence provided does not")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The given sentence does not contain any explicit bias.",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence above can be classified as potentially biased",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence provided contains some elements that could potentially indicate bias",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above can be classified as biased.",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above can be classified as NOT BIASED.",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            'The sentence "He did not elaborate" is a neutral statement that does not show any bias.',
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The given sentence does not appear to contain any explicit bias",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_openai_gpt_3_5_turbo["label"] = df_openai_gpt_3_5_turbo.apply(
    update_label,
    axis=1,
)
df_openai_gpt_3_5_turbo.query("label == '?'")


# In[27]:


# manual assignment
# if there is a tendency like 'may be biased/potential bias' or 'is factual' we classified it as biased or not biased respectively
df_openai_gpt_3_5_turbo.loc[
    [
        1030,
        1560,
        1566,
        1631,
        2057,
        2069,
        2342,
        2693,
        3024,
        3120,
        3342,
        3555,
    ],
    "label",
] = "BIASED"
df_openai_gpt_3_5_turbo.loc[
    [
        37,
        75,
        187,
        249,
        476,
        513,
        576,
        581,
        619,
        640,
        642,
        660,
        681,
        804,
        805,
        929,
        1068,
        1074,
        1084,
        1109,
        1275,
        1300,
        1398,
        1475,
        1615,
        1660,
        1689,
        1707,
        1771,
        1809,
        1828,
        1907,
        1911,
        1915,
        2017,
        2123,
        2177,
        2270,
        2272,
        2282,
        2712,
        2751,
        2798,
        2942,
        2954,
        2983,
        3028,
        3077,
        3094,
        3224,
        3249,
        3299,
        3391,
        3457,
        3466,
        3536,
        3676,
        3739,
        3850,
        3859,
        3917,
        1751,
        1878,
        2130,
        2349,
        2525,
        2553,
        2664,
        2715,
        2782,
        3726,
        3943,
    ],
    "label",
] = "NOT BIASED"

# undecidablle 24, 200, 483 859 900 914 1016, 3018, 3234, 3521, 3603, 3732
# mapped it to NOT BIASED  if "no obvious sign of bias"
# often ends with stuff like "However, to make a definitive classification, it would be necessary to analyze the broader context, such as the source of the information or any potential framing that may be present.'...

# df_openai_gpt_3_5_turbo.loc[3943]['response']
df_openai_gpt_3_5_turbo.query("label == '?'")


# In[28]:


df_openai_gpt_3_5_turbo = df_openai_gpt_3_5_turbo.rename(
    columns={"label": "gpt_3_5_label"},
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
    df_openai_gpt_3_5_turbo[["text", "gpt_3_5_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
gpt_3_5_label = df_merge["gpt_3_5_label"].astype(int)


# In[29]:


print(
    "F1-Score with GPT 3.5 Turbo with (0 shot CoT): ",
    f1_score(ground_truth, gpt_3_5_label),
)
print(
    "Precision with GPT 3.5 Turbo with (0 shot CoT): ",
    precision_score(ground_truth, gpt_3_5_label),
)
print(
    "Recall with GPT 3.5 Turbo with (0 shot CoT): ",
    recall_score(ground_truth, gpt_3_5_label),
)
print(
    "Accuracy with GPT 3.5 Turbo with (0 shot CoT): ",
    accuracy_score(ground_truth, gpt_3_5_label),
)


# # GPT 4 turbo

# In[30]:


df_openai_gpt_4_turbo.query("label == '?'")


# In[31]:


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
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    else:
        return row["label"]


df_openai_gpt_4_turbo["label"] = df_openai_gpt_4_turbo.apply(
    update_label,
    axis=1,
)
df_openai_gpt_4_turbo.query("label == '?'")


# In[32]:


df_openai_gpt_4_turbo.loc[1].response


# In[33]:


df_openai_gpt_4_turbo.loc[[], "label"] = "BIASED"
df_openai_gpt_4_turbo.loc[
    [42, 29, 28, 25, 21, 17, 13, 8, 3, 1],
    "label",
] = "NOT BIASED"


df_openai_gpt_4_turbo.loc[
    [
        48,
        46,
        39,
        38,
        37,
        33,
        32,
        30,
        20,
        19,
        18,
        16,
        9,
        7,
        5,
    ],
    "label",
] = "?"

# tends to do if... then BIASED, if ... then NOT BIASED argumentation -> here uncertain

# Mostly does Step 1:... Step 2:.... . Often this chain is to long for the max new tokens.
# often cuted directly before the answer
# Also often Phrases like "Please provide your answer in the format specified." backchecking for facts -> due to chat model?

# Steps (aprox Summary):
# 1. Identify the language used in the sentence
# 2. Analyze the language used in the sentence
# 3. Consider the context of the sentence
# 4. Determine the bias of the sentence


# In[34]:


df_openai_gpt_4_turbo = df_openai_gpt_4_turbo.rename(
    columns={"label": "gpt_4_label"},
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
# df_merge_all_runs = df_merge_all_runs.merge(df_openai_gpt_4_turbo[df_openai_gpt_4_turbo['gpt_4_label'] != '?'][['text', 'gpt_4_label']], on='text')
# df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(df_openai_gpt_4_turbo[['text', 'gpt_4_label']], on='text')


ground_truth = df_merge["label"].astype(int)
gpt_4_label = df_merge["gpt_4_label"].astype(int)


# In[35]:


print(
    "F1-Score with GPT 4 turbo with (0 shot CoT): ",
    f1_score(ground_truth, gpt_4_label),
)
print(
    "Precision with GPT 4 turbo with (0 shot CoT): ",
    precision_score(ground_truth, gpt_4_label),
)
print(
    "Recall with GPT 4 turbo with (0 shot CoT ): ",
    recall_score(ground_truth, gpt_4_label),
)
print(
    "Accuracy with GPT 4 turbo with (0 shot CoT): ",
    accuracy_score(ground_truth, gpt_4_label),
)


# # Llama-2-7b-chat-hf

# In[36]:


df_Llama_2_7b.query("label == '?'")


# In[37]:


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
        row["response"].startswith("The statement is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is labeled as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence appears to be NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].endswith("is NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("is BIASED.") and row["label"] == "?":
        return "BIASED"
    elif (
        "can be classified as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "classify this sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify this sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "the sentence is NOT BIASED" in row["response"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "sentence is NOT biased" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "classified as SUBTLY BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif "appears to be NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif "sentence is BIASED" in row["response"] and row["label"] == "?":
        return "BIASED"
    elif "sentence is NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("BIASED.") and row["label"] == "?":
        return "BIASED"
    elif row["response"].endswith("NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].endswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].endswith("Please provide your answer.")
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith("Please provide your answer for each step.")
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please provide your answer and explain your reasoning.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "What is your classification of the sentence?",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please provide your answers for each step, and I will let you know if you are correct or not.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please provide your answer and reasoning for each point.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please provide your answer in the format specified.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith("Please provide your answer for each point.")
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please select one of the options from the table above.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith("Please provide your answer and reasoning.")
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please answer these questions and I will provide you with the classification of the sentence.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith("Please explain your reasoning.")
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please provide your answer for each point and I will give you the final classification.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith("Please select one of the options above.")
        and row["label"] == "?"
    ):
        return "!"

    else:
        return row["label"]


df_Llama_2_7b["label"] = df_Llama_2_7b.apply(update_label, axis=1)
df_Llama_2_7b.query("label == '?'")


# In[38]:


df_Llama_2_7b["response"].loc[1772]


# In[39]:


df_Llama_2_7b.loc[
    [
        16,
        4097,
        4111,
        4074,
        4111,
        4074,
        199,
        228,
        506,
        516,
        3792,
        552,
        608,
        635,
        707,
        970,
        1721,
        2601,
        2445,
        2444,
        2402,
        2384,
        2207,
        1973,
        1918,
        1842,
    ],
    "label",
] = "BIASED"
df_Llama_2_7b.loc[
    [
        4028,
        3819,
        693,
        907,
        1054,
        1267,
        1326,
        2584,
        2202,
        2143,
        2134,
    ],
    "label",
] = "NOT BIASED"


df_Llama_2_7b.loc[
    [
        68,
        4085,
        4086,
        4054,
        4051,
        169,
        992,
        995,
        155,
        4054,
        190,
        25,
        87,
        214,
        249,
        253,
        256,
        3948,
        263,
        304,
        362,
        364,
        381,
        3910,
        3920,
        3969,
        4009,
        4021,
        393,
        410,
        411,
        435,
        444,
        458,
        485,
        491,
        502,
        3826,
        3832,
        3872,
        3894,
        3906,
        505,
        510,
        515,
        3801,
        3804,
        546,
        565,
        588,
        612,
        623,
        666,
        698,
        709,
        713,
        3738,
        3760,
        720,
        747,
        770,
        773,
        3710,
        3713,
        3761,
        3772,
        3790,
        797,
        810,
        820,
        831,
        867,
        3615,
        3651,
        3668,
        3676,
        3689,
        906,
        948,
        955,
        3548,
        3571,
        3575,
        3594,
        3608,
        942,
        967,
        978,
        988,
        1029,
        1036,
        1038,
        1039,
        3480,
        3484,
        3526,
        3532,
        3545,
        1090,
        1111,
        1122,
        1122,
        1129,
        1180,
        3352,
        3400,
        3402,
        3409,
        3425,
        1187,
        1200,
        1201,
        1214,
        1215,
        3331,
        3340,
        3347,
        3349,
        3350,
        1257,
        1273,
        1315,
        3241,
        3269,
        3298,
        3315,
        3322,
        1335,
        1343,
        1351,
        1356,
        1376,
        3162,
        3185,
        3190,
        3205,
        3228,
        1406,
        1413,
        1418,
        1426,
        1438,
        3089,
        3109,
        3121,
        3123,
        3151,
        1442,
        1449,
        1453,
        1460,
        1468,
        1474,
        1506,
        1538,
        1554,
        3019,
        3020,
        3023,
        3047,
        3082,
        1591,
        1627,
        1651,
        1663,
        1668,
        2949,
        2960,
        2962,
        2983,
        2988,
        1709,
        1716,
        1736,
        1768,
        2850,
        2877,
        2885,
        2934,
        2936,
        2824,
        2789,
        2757,
        2751,
        2749,
        2733,
        2705,
        2684,
        2679,
        2643,
        2638,
        2620,
        2617,
        2610,
        2550,
        2544,
        2539,
        2538,
        2522,
        2514,
        2345,
        2342,
        2339,
        2298,
        2273,
        2229,
        2221,
        2205,
        2172,
        2129,
        2127,
        2120,
        2115,
        2089,
        2062,
        2052,
        2042,
        2035,
        2025,
        1986,
        1980,
        1919,
        1907,
        1869,
        1853,
        1784,
        1772,
    ],
    "label",
] = "!"

# tends to do if... then BIASED, if ... then NOT BIASED argumentation -> here uncertain

# Mostly does Step 1:... Step 2:.... . Often this chain is to long for the max new tokens.
# often cuted directly before the answer
# Also often Phrases like "Please provide your answer in the format specified." backchecking for facts -> due to chat model?

# Steps (aprox Summary):
# 1. Identify the language used in the sentence
# 2. Analyze the language used in the sentence
# 3. Consider the context of the sentence
# 4. Determine the bias of the sentence
df_Llama_2_7b.query("label == '?'")


# In[40]:


# replace intermediat "!" flag to "?"
df_Llama_2_7b["label"] = df_Llama_2_7b["label"].replace("!", "?")

df_Llama_2_7b = df_Llama_2_7b.rename(columns={"label": "llama_7b_label"})
df_Llama_2_7b["llama_7b_label"] = df_Llama_2_7b["llama_7b_label"].replace(
    "BIASED",
    1,
)
df_Llama_2_7b["llama_7b_label"] = df_Llama_2_7b["llama_7b_label"].replace(
    "NOT BIASED",
    0,
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
    df_Llama_2_7b[["text", "llama_7b_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
llama_7b_label = df_merge["llama_7b_label"].astype(int)


# In[41]:


print(
    "F1-Score with llama 7b (0 Shot CoT): ",
    f1_score(ground_truth, llama_7b_label),
)
print(
    "Precision with llama 7b (0 Shot CoT): ",
    precision_score(ground_truth, llama_7b_label),
)
print(
    "Recall with llama 7b (0 Shot CoT): ",
    recall_score(ground_truth, llama_7b_label),
)
print(
    "Accuracy with llama 7b (0 Shot CoT): ",
    accuracy_score(ground_truth, llama_7b_label),
)


# # Llama-2-13b-chat-hf

# In[42]:


df_Llama_2_13b.query("label == '?'")


# In[43]:


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
        row["response"].startswith("The statement is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is labeled as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence appears to be NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].endswith("is NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("is BIASED.") and row["label"] == "?":
        return "BIASED"
    elif (
        "can be classified as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "classify this sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify this sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "the sentence is NOT BIASED" in row["response"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "sentence is NOT biased" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "classified as SUBTLY BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif "appears to be NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif "sentence is BIASED" in row["response"] and row["label"] == "?":
        return "BIASED"
    elif "sentence is NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("BIASED.") and row["label"] == "?":
        return "BIASED"
    elif row["response"].endswith("NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif row["response"].endswith("NOT BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].endswith("Please provide your answer.")
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith("Please provide your answer for each step.")
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please provide your answer and explain your reasoning.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "What is your classification of the sentence?",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please provide your answers for each step, and I will let you know if you are correct or not.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please answer the questions above before I can help you classify the sentence.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "answers to these questions before I can classify the sentence"
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "answer the questions above before I can classify the sentence"
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "answer the questions before I can classify the sentence"
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Please choose your answer." in row["response"] and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please answer the questions before I can classify the sentence as biased or not biased.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith("BIASED or NOT BIASED?")
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "Please answer the questions above before I can give you the answer to your question.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "I'll wait for your response before proceeding further.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Please provide your answer" in row["response"] and row["label"] == "?"
    ):
        return "!"
    elif (
        "Please answer the question" in row["response"] and row["label"] == "?"
    ):
        return "!"
    elif (
        "biased" not in row["response"].lower()
        and "not biased" not in row["response"].lower()
    ) and row["label"] == "?":
        return "!"
    else:
        return row["label"]


df_Llama_2_13b["label"] = df_Llama_2_13b.apply(update_label, axis=1)
df_Llama_2_13b.query("label == '?'")


# In[44]:


df_Llama_2_13b["response"].loc[1827]


# In[45]:


df_Llama_2_13b.loc[
    [
        332,
        3792,
        687,
        697,
        1008,
        3888,
        3290,
        3266,
        3234,
        3005,
        2714,
        2093,
    ],
    "label",
] = "BIASED"
df_Llama_2_13b.loc[
    [
        472,
        813,
        838,
        3782,
        1106,
        1360,
        1472,
        3537,
        3540,
        3541,
        3550,
        3561,
        1685,
        1799,
        3332,
        3167,
        2692,
        2461,
        2254,
        1867,
        1853,
        1850,
        1827,
    ],
    "label",
] = "NOT BIASED"

df_Llama_2_13b.loc[
    [
        4017,
        3999,
        3,
        37,
        66,
        127,
        166,
        197,
        244,
        286,
        304,
        304,
        312,
        315,
        366,
        385,
        615,
        720,
        789,
        805,
        861,
        867,
        936,
        945,
        978,
        997,
        1031,
        1036,
        3833,
        1045,
        1072,
        1178,
        3719,
        3730,
        3745,
        3891,
        3946,
        1370,
        1384,
        1445,
        1478,
        3600,
        3612,
        3634,
        3698,
        3704,
        1523,
        1537,
        1632,
        1641,
        1669,
        1768,
        1813,
        1825,
        3432,
        3474,
        3499,
        3500,
        3522,
        3402,
        3400,
        3265,
        3206,
        3201,
        3184,
        3150,
        3149,
        3144,
        3076,
        3072,
        2949,
        2934,
        2745,
        2718,
        2705,
        2648,
        2627,
        2618,
        2607,
        2559,
        2526,
        2514,
        2493,
        2469,
        2466,
        2450,
        2395,
        2378,
        2368,
        2292,
        2290,
        2286,
        2281,
        2251,
        2207,
        2162,
        2476,
        2068,
        2030,
        1999,
        1908,
        1907,
        1863,
        1848,
    ],
    "label",
] = "!"


# nice example 2093

# Often asks the about more context or about the opinion of the user. -> Not usable in this scenario

# tends to do if... then BIASED, if ... then NOT BIASED argumentation -> here uncertain
# Reasoning way:
# Step 1: Who is the speaker?
# Step 2: Who is the subject of the sentence?
# Step 3: What is the tone of the sentence?
# Step 4: What is the purpose of the sentence?
# Step 5: Is the sentence factual or opinion-based?
# Step 6: Is the sentence biased or not biased?
df_Llama_2_13b.query("label == '?'")


# In[46]:


# map "!" flag back to "?"
df_Llama_2_13b["label"] = df_Llama_2_13b["label"].replace("!", "?")

df_Llama_2_13b = df_Llama_2_13b.rename(columns={"label": "llama_13b_label"})
df_Llama_2_13b["llama_13b_label"] = df_Llama_2_13b["llama_13b_label"].replace(
    "BIASED",
    1,
)
df_Llama_2_13b["llama_13b_label"] = df_Llama_2_13b["llama_13b_label"].replace(
    "NOT BIASED",
    0,
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
    df_Llama_2_13b[["text", "llama_13b_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
llama_13b_label = df_merge["llama_13b_label"].astype(int)


# In[47]:


print(
    "F1-Score with Llama 2 13b with (0 shot CoT): ",
    f1_score(ground_truth, llama_13b_label),
)
print(
    "Precision with Llama 2 13b with (0 shot CoT): ",
    precision_score(ground_truth, llama_13b_label),
)
print(
    "Recall with Llama 2 13b with (0 shot CoT): ",
    recall_score(ground_truth, llama_13b_label),
)
print(
    "Accuracy with Llama 2 13b with (0 shot CoT): ",
    accuracy_score(ground_truth, llama_13b_label),
)


# # Mistral-7B-Instruct-v0.1

# In[48]:


df_mistral_7b.query("label == '?'")


# In[49]:


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
        row["response"].startswith("The statement is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is labeled as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence appears to be NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].endswith("is NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("is BIASED.") and row["label"] == "?":
        return "BIASED"
    elif (
        "can be classified as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "classify this sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify this sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "the sentence is NOT BIASED" in row["response"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "sentence is NOT biased" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "classified as SUBTLY BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif "appears to be NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("BIASED.") and row["label"] == "?":
        return "BIASED"
    elif row["response"].endswith("NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "biased" not in row["response"].lower()
        and "not biased" not in row["response"].lower()
    ) and row["label"] == "?":
        return "!"
    elif (
        row["response"].endswith(
            "Can you provide more context or clarify the author's opinion?",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        row["response"].endswith(
            "cannot classify the sentence as biased or not biased without more context.",
        )
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "cannot classify the sentence as biased or not biased"
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "it is impossible to classify the sentence as biased or not biased."
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Without more context, it is difficult to determine whether the sentence is biased or not."
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Without more context, it is difficult to determine whether the sentence is BIASED or NOT BIASED."
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "cannot be classified as either biased or not biased without additional context."
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "cannot be classified as biased or not biased" in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "the sentence is biased because" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "Overall, the sentence is not biased" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "Overall, the sentence is biased" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "cannot definitively classify the sentence as biased or not biased"
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "cannot be classified as BIASED or NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "cannot classify it as BIASED or NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "cannot definitively classify the sentence as either BIASED or NOT BIASED"
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "Overall, the sentence is a mix of both biased and unbiased statements"
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "could be classified as BIASED or NOT BIASED depending on the context"
        in row["response"]
        and row["label"] == "?"
    ):
        return "!"
    elif (
        "we cannot definitively classify it as biased or not biased"
        in row["response"].lower()
        and row["label"] == "?"
    ):
        return "!"
    else:
        return row["label"]


df_mistral_7b["label"] = df_mistral_7b.apply(update_label, axis=1)
df_mistral_7b.query("label == '?'")


# In[50]:


df_mistral_7b.response.loc[2171]


# In[51]:


df_mistral_7b.loc[
    [
        320,
        535,
        4019,
        845,
        897,
        1093,
        3799,
        1728,
        1858,
        1972,
        2004,
        2061,
        2732,
        2333,
        2273,
    ],
    "label",
] = "BIASED"
df_mistral_7b.loc[
    [
        4,
        75,
        75,
        352,
        425,
        511,
        665,
        804,
        965,
        1038,
        1053,
        3709,
        1548,
        1713,
        1745,
        1831,
        1832,
        3234,
        3285,
        1866,
        1871,
        3105,
        3058,
        2895,
        2884,
        2693,
        2645,
        2413,
        2250,
        2203,
    ],
    "label",
] = "NOT BIASED"

df_mistral_7b.loc[
    [
        72,
        194,
        4,
        194,
        291,
        329,
        333,
        4031,
        4084,
        4096,
        4106,
        4109,
        457,
        467,
        469,
        518,
        3974,
        524,
        528,
        542,
        545,
        3935,
        3956,
        4018,
        4021,
        575,
        593,
        601,
        630,
        647,
        659,
        680,
        685,
        767,
        768,
        792,
        796,
        882,
        901,
        933,
        1030,
        1075,
        1090,
        3871,
        3885,
        3886,
        1121,
        1131,
        1153,
        1173,
        3742,
        3744,
        3773,
        3792,
        3803,
        1125,
        1192,
        1205,
        1247,
        1261,
        3702,
        3703,
        3720,
        3727,
        1281,
        1288,
        1304,
        1309,
        1324,
        3627,
        3635,
        3638,
        3655,
        1365,
        1370,
        1371,
        1380,
        1385,
        3591,
        3608,
        1398,
        1443,
        1462,
        1468,
        1506,
        1520,
        1558,
        1559,
        1574,
        1593,
        3558,
        3577,
        3624,
        3626,
        3666,
        1600,
        1651,
        1709,
        3464,
        3475,
        3477,
        3496,
        3522,
        1737,
        1751,
        1790,
        1815,
        3364,
        3383,
        3393,
        3449,
        3454,
        1835,
        1848,
        1856,
        3250,
        3263,
        3337,
        1886,
        1892,
        3112,
        3153,
        3163,
        3206,
        3223,
        1927,
        1975,
        3080,
        3088,
        3095,
        3098,
        2072,
        2075,
        2112,
        2115,
        2138,
        3013,
        3026,
        3037,
        3079,
        3012,
        3010,
        2999,
        2985,
        2947,
        2939,
        2925,
        2916,
        2889,
        2881,
        2877,
        2874,
        2836,
        2828,
        2797,
        2786,
        2775,
        2753,
        2747,
        2699,
        2674,
        2662,
        2661,
        2651,
        2641,
        2634,
        2619,
        2571,
        2533,
        2524,
        2483,
        2479,
        2458,
        2440,
        2436,
        2427,
        2397,
        2396,
        2393,
        2371,
        2353,
        2349,
        2230,
        2221,
        2215,
        2201,
        2171,
    ],
    "label",
] = "!"

# tends to do if... then BIASED, if ... then NOT BIASED argumentation -> here uncertain
# also often asks for context or says the BIAS is context dependent.
# Often if classified as BIASED -> dependend on word choice (same as original BABE ranking)

# Process:
# 1. First, we need to identify the source of the information. Is it a reputable source or a biased one?
# 2. Next, we need to analyze the language used in the sentence. Is it objective or subjective?
# 3. Finally, we need to consider the context in which the sentence is used. Is it being used to present facts or to make an argument?
# nBased on these criteria, we can classify the sentence as either BIASED or NOT BIASED.
df_mistral_7b.query("label == '?'")


# In[52]:


# replace intermediat "!" flag to "?"
df_mistral_7b["label"] = df_mistral_7b["label"].replace("!", "?")

df_mistral_7b = df_mistral_7b.rename(columns={"label": "mistral_7b_label"})
df_mistral_7b["mistral_7b_label"] = df_mistral_7b["mistral_7b_label"].replace(
    "BIASED",
    1,
)
df_mistral_7b["mistral_7b_label"] = df_mistral_7b["mistral_7b_label"].replace(
    "NOT BIASED",
    0,
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
    df_mistral_7b[["text", "mistral_7b_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
df_mistral_7b_label = df_merge["mistral_7b_label"].astype(int)


# In[53]:


print(
    "F1-Score with Mistral-7B-Instruct-v0.1 with (0 shot CoT): ",
    f1_score(ground_truth, df_mistral_7b_label),
)
print(
    "Precision with Mistral-7B-Instruct-v0.1 with (0 shot CoT): ",
    precision_score(ground_truth, df_mistral_7b_label),
)
print(
    "Recall with Mistral-7B-Instruct-v0.1 with (0 shot CoT): ",
    recall_score(ground_truth, df_mistral_7b_label),
)
print(
    "Accuracy with Mistral-7B-Instruct-v0.1 with (0 shot CoT): ",
    accuracy_score(ground_truth, df_mistral_7b_label),
)


# # Mixtral-8x7B

# In[54]:


df_mixtral_8x7b.query("label == '?'")


# In[55]:


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
        row["response"].startswith("The statement is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is labeled as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence appears to be NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].endswith("is NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("is BIASED.") and row["label"] == "?":
        return "BIASED"
    elif (
        "can be classified as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "classify this sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify this sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "the sentence is NOT BIASED" in row["response"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "sentence is NOT biased" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "classified as SUBTLY BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif "appears to be NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("BIASED.") and row["label"] == "?":
        return "BIASED"
    elif row["response"].endswith("NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif "Verdict: NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif "Verdict: BIASED" in row["response"] and row["label"] == "?":
        return "BIASED"
    elif (
        "Classification: NOT BIASED" in row["response"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "Classification: BIASED" in row["response"] and row["label"] == "?":
        return "BIASED"
    elif "Answer: NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif "Answer: BIASED" in row["response"] and row["label"] == "?":
        return "BIASED"
    else:
        return row["label"]


df_mixtral_8x7b["label"] = df_mixtral_8x7b.apply(update_label, axis=1)
df_mixtral_8x7b.query("label == '?'")


# In[56]:


df_mixtral_8x7b["response"].loc[1139]


# In[57]:


df_mixtral_8x7b.loc[
    [
        3937,
        3843,
        78,
        253,
        253,
        3651,
        750,
        758,
        767,
        801,
        889,
        3323,
        3358,
        992,
        994,
        3135,
        3059,
        2870,
        2455,
        2351,
        2303,
        2280,
        2163,
        1936,
        1806,
        1800,
        1649,
        1568,
        1541,
        1512,
    ],
    "label",
] = "BIASED"
df_mixtral_8x7b.loc[
    [
        247,
        213,
        251,
        315,
        355,
        513,
        610,
        629,
        633,
        702,
        724,
        3409,
        3488,
        848,
        904,
        928,
        948,
        3394,
        965,
        1103,
        1133,
        3132,
        3205,
        3223,
        3080,
        2926,
        2883,
        2863,
        2593,
        2375,
        2289,
        2267,
        2258,
        2188,
        1742,
        1718,
        1470,
        1326,
    ],
    "label",
] = "NOT BIASED"


df_mixtral_8x7b.loc[
    [
        3961,
        4032,
        3933,
        64,
        4,
        436,
        694,
        3694,
        3706,
        3724,
        3769,
        830,
        3398,
        3417,
        3600,
        3377,
        3380,
        3207,
        3108,
        2958,
        2920,
        2798,
        2787,
        2754,
        2714,
        2650,
        2449,
        2328,
        2252,
        2242,
        1979,
        1917,
        1876,
        1796,
        1791,
        1578,
        1570,
        1501,
        1413,
        1384,
        1139,
    ],
    "label",
] = "!"

# tends to do if... then BIASED, if ... then NOT BIASED argumentation -> here uncertain
# undefined 2964, 2630, 29
df_mixtral_8x7b.query("label == '?'")


# In[58]:


# map "!" flag back to "?"
df_mixtral_8x7b["label"] = df_mixtral_8x7b["label"].replace("!", "?")

df_mixtral_8x7b = df_mixtral_8x7b.rename(
    columns={"label": "mixtral_8x7b_label"},
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
    df_mixtral_8x7b[["text", "mixtral_8x7b_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
df_mixtral_8x7b_label = df_merge["mixtral_8x7b_label"].astype(int)


# In[59]:


print(
    "F1-Score with mixtral_8x7b with (0 shot CoT): ",
    f1_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Precision with mixtral_8x7b with (0 shot CoT): ",
    precision_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Recall with mixtral_8x7b with (0 shot CoT): ",
    recall_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Accuracy with mixtral_8x7b with (0 shot CoT): ",
    accuracy_score(ground_truth, df_mixtral_8x7b_label),
)


# # OpenChat_3.5

# In[60]:


df_openchat_3_5.query("label == '?'")


# In[61]:


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
        row["response"].startswith("The statement is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith(
            "The sentence above is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is labeled as not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence appears to be NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].endswith("is NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("is BIASED.") and row["label"] == "?":
        return "BIASED"
    elif (
        "can be classified as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "classify this sentence as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify this sentence as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as NOT BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "the sentence is NOT BIASED" in row["response"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "sentence is NOT biased" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "classified as SUBTLY BIASED" in row["response"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif "appears to be NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].endswith("BIASED.") and row["label"] == "?":
        return "BIASED"
    elif row["response"].endswith("NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif "Verdict: NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif "Verdict: BIASED" in row["response"] and row["label"] == "?":
        return "BIASED"
    elif (
        "Classification: NOT BIASED" in row["response"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "Classification: BIASED" in row["response"] and row["label"] == "?":
        return "BIASED"
    elif "Answer: NOT BIASED" in row["response"] and row["label"] == "?":
        return "NOT BIASED"
    elif "Answer: BIASED" in row["response"] and row["label"] == "?":
        return "BIASED"
    else:
        return row["label"]


df_openchat_3_5["label"] = df_openchat_3_5.apply(update_label, axis=1)
df_openchat_3_5.query("label == '?'")


# In[62]:


df_openchat_3_5.response.loc[2926]


# In[63]:


df_openchat_3_5.loc[
    [
        145,
        3921,
        4,
        3734,
        3790,
        903,
        3372,
        3372,
        3274,
        2938,
        2936,
        2696,
        2867,
        2328,
        2242,
        1791,
    ],
    "label",
] = "BIASED"
df_openchat_3_5.loc[
    [
        142,
        193,
        254,
        3741,
        363,
        795,
        876,
        1045,
        1098,
        3580,
        1161,
        1162,
        3309,
        3328,
        3372,
        3180,
        3169,
        3108,
        3088,
        2911,
        2869,
        2714,
        2713,
        2685,
        2684,
        2291,
        1757,
        1637,
        1462,
        1421,
        1384,
        1346,
        1277,
    ],
    "label",
] = "NOT BIASED"


df_openchat_3_5.loc[
    [
        22,
        182,
        3893,
        3978,
        4044,
        4092,
        199,
        299,
        343,
        361,
        3669,
        3779,
        422,
        428,
        442,
        476,
        3562,
        500,
        523,
        550,
        580,
        754,
        827,
        869,
        932,
        963,
        987,
        1048,
        1064,
        1075,
        3504,
        3591,
        3621,
        3657,
        1144,
        1146,
        1177,
        3354,
        3456,
        3354,
        3265,
        3172,
        3170,
        3084,
        2844,
        2834,
        2827,
        2739,
        2736,
        2719,
        2580,
        2515,
        2423,
        2332,
        2293,
        2216,
        2185,
        2134,
        2066,
        2007,
        1927,
        1758,
        1755,
        1714,
        1706,
        1650,
        1637,
        1609,
        1411,
        1406,
        1187,
        1586,
        2455,
        2926,
    ],
    "label",
] = "!"

df_openchat_3_5.query("label == '?'")


# In[64]:


# map "!" flag back to "?"
df_openchat_3_5["label"] = df_openchat_3_5["label"].replace("!", "?")

df_openchat_3_5 = df_openchat_3_5.rename(columns={"label": "openchat_label"})
df_openchat_3_5["openchat_label"] = df_openchat_3_5["openchat_label"].replace(
    "BIASED",
    1,
)
df_openchat_3_5["openchat_label"] = df_openchat_3_5["openchat_label"].replace(
    "NOT BIASED",
    0,
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
    df_openchat_3_5[["text", "openchat_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
openchat_label = df_merge["openchat_label"].astype(int)


# In[65]:


print(
    "F1-Score with OpenChat 3.5 with (0 shot CoT): ",
    f1_score(ground_truth, openchat_label),
)
print(
    "Precision with OpenChat 3.5 with (0 shot CoT): ",
    precision_score(ground_truth, openchat_label),
)
print(
    "Recall with OpenChat 3.5 with (0 shot CoT): ",
    recall_score(ground_truth, openchat_label),
)
print(
    "Accuracy with OpenChat 3.5 with (0 shot CoT): ",
    accuracy_score(ground_truth, openchat_label),
)


# # zephyr-7b-beta

# In[66]:


df_zephyr_7b_beta["response_short"] = df_zephyr_7b_beta.apply(
    split_and_store,
    axis=1,
)
df_zephyr_7b_beta.query("label == '?'")


# In[67]:


def update_label(row):
    if row["response_short"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif (
        row["response_short"].startswith("NOT BIASED") and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].startswith("Classification: NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].startswith("Classification: BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response_short"].startswith("The statement is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response_short"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].startswith(
            "Classify the sentence above as BIASED or NOT BIASED.\n\nOutput: BIASED",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response_short"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response_short"].startswith("The sentence is BIASED")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response_short"].startswith(
            "The sentence above is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].startswith(
            "The sentence is classified as not biased",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].startswith(
            "The sentence is labeled as not biased",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].startswith(
            "The sentence appears to be NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response_short"].endswith("is NOT BIASED.")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response_short"].endswith("is BIASED.") and row["label"] == "?":
        return "BIASED"
    elif (
        "can be classified as NOT BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as NOT BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify the sentence as BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "classify this sentence as NOT BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classify this sentence as BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "I would classify it as NOT BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "the sentence is NOT BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "sentence is NOT biased" in row["response_short"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "classified as SUBTLY BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        "appears to be NOT BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response_short"].endswith("BIASED.") and row["label"] == "?":
        return "BIASED"
    elif row["response_short"].endswith("NOT BIASED.") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        "Verdict: NOT BIASED" in row["response_short"] and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif "Verdict: BIASED" in row["response_short"] and row["label"] == "?":
        return "BIASED"
    elif (
        "Classification: NOT BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "Classification: BIASED" in row["response_short"]
        and row["label"] == "?"
    ):
        return "BIASED"
    elif "Answer: NOT BIASED" in row["response_short"] and row["label"] == "?":
        return "NOT BIASED"
    elif "Answer: BIASED" in row["response_short"] and row["label"] == "?":
        return "BIASED"
    elif "Output: NOT BIASED" in row["response_short"] and row["label"] == "?":
        return "NOT BIASED"
    elif "Output: BIASED" in row["response_short"] and row["label"] == "?":
        return "BIASED"
    elif (
        "NOT BIASED".lower() not in row["response_short"].lower()
        and "BIASED".lower() not in row["response_short"].lower()
    ) and row["label"] == "?":
        return "!"
    elif (
        "Overall, the sentence is NOT BIASED".lower()
        in row["response_short"].lower()
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        "Overall, the sentence is BIASED".lower()
        in row["response_short"].lower()
        and row["label"] == "?"
    ):
        return "BIASED"

    else:
        return row["label"]


df_zephyr_7b_beta["label"] = df_zephyr_7b_beta.apply(update_label, axis=1)
df_zephyr_7b_beta.query("label == '?'")


# In[68]:


df_zephyr_7b_beta["response_short"].loc[1447]


# In[69]:


# manual assignment
df_zephyr_7b_beta.loc[
    [
        182,
        198,
        4059,
        4071,
        307,
        337,
        357,
        395,
        411,
        426,
        465,
        474,
        502,
        654,
        836,
        841,
        854,
        856,
        892,
        981,
        983,
        3866,
        3887,
        3913,
        1186,
        3705,
        3738,
        1191,
        1196,
        1234,
        3636,
        3661,
        1310,
        1316,
        1319,
        1325,
        1390,
        1475,
        3352,
        1696,
        1698,
        3326,
        1716,
        3148,
        3068,
        3032,
        2557,
        2501,
        2424,
        2375,
        2263,
        2113,
        2059,
        2053,
        2045,
        2022,
        1997,
        1976,
        1447,
    ],
    "label",
] = "BIASED"
df_zephyr_7b_beta.loc[
    [
        74,
        4048,
        4007,
        639,
        746,
        787,
        812,
        843,
        1065,
        1075,
        4015,
        1134,
        3744,
        1231,
        3658,
        3469,
        1653,
        3135,
        3278,
        2517,
        2339,
        2233,
        2071,
        2017,
        1927,
    ],
    "label",
] = "NOT BIASED"

# not assignable e.g. UNCLEAR, MIXED, BORDERLINE, NOT APPLICABLE or bad response
df_zephyr_7b_beta.loc[
    [
        115,
        160,
        198,
        4063,
        4100,
        203,
        220,
        393,
        424,
        3935,
        3957,
        3993,
        457,
        515,
        557,
        582,
        591,
        602,
        642,
        685,
        697,
        700,
        730,
        741,
        816,
        822,
        886,
        936,
        953,
        3769,
        1078,
        1095,
        1142,
        1165,
        1183,
        3728,
        3757,
        1190,
        3625,
        3700,
        1270,
        3512,
        3513,
        3586,
        3590,
        3600,
        1391,
        1399,
        1476,
        3366,
        3409,
        3508,
        1599,
        1695,
        3287,
        3292,
        3318,
        3321,
        1735,
        1745,
        1787,
        1790,
        3228,
        3236,
        3132,
        3067,
        3059,
        3043,
        2999,
        2923,
        2912,
        2903,
        2874,
        2808,
        2789,
        2778,
        2741,
        2736,
        2722,
        2714,
        2701,
        2644,
        2640,
        2617,
        2545,
        2519,
        2444,
        2427,
        2406,
        2395,
        2281,
        2243,
        2235,
        2224,
        2182,
        2163,
        2159,
        2097,
        2023,
        1968,
        1956,
        1907,
        1899,
        1841,
    ],
    "label",
] = "!"

# also contains mixed BIASED not BIASED examples
# Often also argues for both sides -> no clear decision -> '!'
# argues for bias and against it.
# Often rates the parts indiviuell for BIAS. Often not the overall rating


# In[70]:


# map "!" flag back to "?"
df_zephyr_7b_beta["label"] = df_zephyr_7b_beta["label"].replace("!", "?")

df_zephyr_7b_beta = df_zephyr_7b_beta.rename(columns={"label": "zephyr_label"})
df_zephyr_7b_beta["zephyr_label"] = df_zephyr_7b_beta["zephyr_label"].replace(
    "BIASED",
    1,
)
df_zephyr_7b_beta["zephyr_label"] = df_zephyr_7b_beta["zephyr_label"].replace(
    "NOT BIASED",
    0,
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
    df_zephyr_7b_beta[["text", "zephyr_label"]],
    on="text",
)

ground_truth = df_merge["label"].astype(int)
zephyr_label = df_merge["zephyr_label"].astype(int)


# In[71]:


print(
    "F1-Score with zephyr beta (0 shot CoT): ",
    f1_score(ground_truth, zephyr_label),
)
print(
    "Precision with zephyr beta (0 shot CoT): ",
    precision_score(ground_truth, zephyr_label),
)
print(
    "Recall with zephyr beta (0 shot CoT): ",
    recall_score(ground_truth, zephyr_label),
)
print(
    "Accuracy with zephyr beta (0 shot CoT): ",
    accuracy_score(ground_truth, zephyr_label),
)


# In[72]:


# safe the file
df_merge_all_runs_with_errors.to_csv("./all_runs_with_errors.csv", index=False)
