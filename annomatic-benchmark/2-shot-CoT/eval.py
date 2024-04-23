# %%
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


# %%

# load

# load results
df_falcon_7b = pd.read_csv(f"./data/falcon-7b-instruct.csv")
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


# %% [markdown]
# # Falcon 7B

# %%
df_falcon_7b.query("label == '?'")


# %%
# preprocessing
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
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_falcon_7b["label"] = df_falcon_7b.apply(update_label, axis=1)
df_falcon_7b.query("label == '?'")

# %%
df_falcon_7b.loc[
    [
        84,
        127,
        129,
        3831,
        3858,
        3906,
        3984,
        4009,
        3703,
        3700,
        3600,
        3585,
        261,
        236,
        3428,
        3406,
        3399,
        3287,
        2920,
        2882,
        2692,
        2557,
        2547,
        2263,
        2116,
        2032,
        1975,
        1835,
        1817,
        1485,
        1354,
        1335,
        1279,
        1195,
        988,
        912,
        764,
        584,
    ],
    "label",
] = "BIASED"
df_falcon_7b.loc[
    [
        18,
        42,
        3768,
        299,
        188,
        179,
        3530,
        3499,
        3355,
        3271,
        3268,
        3235,
        3126,
        3117,
        3074,
        3051,
        3026,
        3019,
        2911,
        2773,
        2746,
        2687,
        2484,
        2234,
        2118,
        2067,
        1771,
        1743,
        1369,
        1354,
        1285,
        961,
        918,
        715,
        633,
        454,
        330,
        314,
    ],
    "label",
] = "NOT BIASED"

# undefined 2083, 716

df_falcon_7b.loc[314]["response"]
df_falcon_7b.query("label == '?'")

# %%
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

# %%
print(
    "F1-Score with Falcon 7b with (2 shot CoT): ",
    f1_score(ground_truth, falcon_7b_label),
)
print(
    "Precision with Falcon 7b with (2 shot CoT): ",
    precision_score(ground_truth, falcon_7b_label),
)
print(
    "Recall with Falcon 7b with (2 shot CoT): ",
    recall_score(ground_truth, falcon_7b_label),
)
print(
    "Accuracy with Falcon 7b with (2 shot CoT): ",
    accuracy_score(ground_truth, falcon_7b_label),
)

# %% [markdown]
# # Flan UL2

# %%
df_flan_ul2.query("label == '?'")


# %%
def update_label(row):
    if row["response"].startswith("Not biased") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].startswith("Not Biased") and row["label"] == "?":
        return "NOT BIASED"
    elif row["response"].startswith("Not BIASED") and row["label"] == "?":
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is neutral")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif row["response"].startswith("BIASED") and row["label"] == "?":
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence presents factual information")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is classified as NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_flan_ul2["label"] = df_flan_ul2.apply(update_label, axis=1)
df_flan_ul2.query("label == '?'")

# %%
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

# %%
print(
    "F1-Score with Flan UL2 (8 shot): ",
    f1_score(ground_truth, flan_ul2_label),
)
print(
    "Precision with Flan UL2 (8 shot): ",
    precision_score(ground_truth, flan_ul2_label),
)
print(
    "Recall with Flan UL2 (8 shot): ",
    recall_score(ground_truth, flan_ul2_label),
)
print(
    "Accuracy with Flan UL2 (8 shot): ",
    accuracy_score(ground_truth, flan_ul2_label),
)

# %% [markdown]
# # GPT-3.5-turbo

# %%
df_openai_gpt_3_5_turbo.query("label == '?'")


# %%
def update_label(row):
    if (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence presents factual")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence provides factual")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("This sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence above is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is classified as NOT BIASED",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is considered not biased",
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

# %%
# manual assignment
df_openai_gpt_3_5_turbo.loc[[3351], "label"] = "BIASED"
df_openai_gpt_3_5_turbo.loc[
    [
        75,
        161,
        263,
        333,
        545,
        564,
        639,
        924,
        971,
        975,
        1074,
        1077,
        1094,
        1551,
        1654,
        1684,
        1757,
        1846,
        1884,
        1886,
        2073,
        2119,
        2176,
        2447,
        2628,
        2748,
        2780,
        2798,
        2953,
        2967,
        3050,
        3128,
        3159,
        3184,
        3206,
        3239,
        3248,
        3295,
        3350,
        3372,
        3384,
        3401,
        3432,
        3443,
        3457,
        3530,
        3602,
        3621,
        3648,
        3904,
        3974,
        3990,
        4011,
    ],
    "label",
] = "NOT BIASED"

# NOT DETERMINABLE 128, 389, 401, 477, 533, 942, 1068, 1428, 1617, 1848, 2004, 2448, 2630, 2936, 2983, 3018, 3127, 3404, 3560

df_openai_gpt_3_5_turbo.loc[4011]["response"]
df_openai_gpt_3_5_turbo.query("label == '?'")

# %%
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

# %%
print(
    "F1-Score with GPT 3.5 Turbo with (2 shot CoT): ",
    f1_score(ground_truth, gpt_3_5_label),
)
print(
    "Precision with GPT 3.5 Turbo with (2 shot CoT): ",
    precision_score(ground_truth, gpt_3_5_label),
)
print(
    "Recall with GPT 3.5 Turbo with (2 shot CoT): ",
    recall_score(ground_truth, gpt_3_5_label),
)
print(
    "Accuracy with GPT 3.5 Turbo with (2 shot CoT): ",
    accuracy_score(ground_truth, gpt_3_5_label),
)

# %%
df_openai_gpt_4_turbo.query("label == '?'")


# %%
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
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_openai_gpt_4_turbo["label"] = df_openai_gpt_4_turbo.apply(
    update_label,
    axis=1,
)
df_openai_gpt_4_turbo.query("label == '?'")

# %%
df_openai_gpt_4_turbo.loc[
    [
        4,
        3963,
        103,
        3841,
        3787,
        3730,
        3627,
        2903,
        911,
        2863,
        2751,
        2711,
        2433,
        2292,
        1964,
        1704,
        1481,
        1262,
        1082,
    ],
    "label",
] = "BIASED"
df_openai_gpt_4_turbo.loc[
    [
        3999,
        3906,
        12,
        37,
        173,
        3845,
        3682,
        3679,
        267,
        263,
        246,
        185,
        285,
        291,
        3501,
        3533,
        3601,
        412,
        438,
        474,
        3198,
        3297,
        3327,
        3390,
        525,
        534,
        545,
        3037,
        3047,
        3113,
        3179,
        627,
        663,
        2993,
        3008,
        3022,
        799,
        743,
        735,
        860,
        861,
        863,
        2828,
        2844,
        2854,
        898,
        1004,
        2765,
        2774,
        2815,
        2694,
        2685,
        2670,
        1068,
        2581,
        2344,
        2116,
        2074,
        2054,
        2023,
        2008,
        1917,
        1897,
        1847,
        1647,
        1490,
        1472,
        1462,
        1289,
        1243,
        1232,
        1229,
        1224,
        1070,
    ],
    "label",
] = "NOT BIASED"

# nicht zuweisbar
df_openai_gpt_4_turbo.loc[
    [
        3984,
        3892,
        29,
        75,
        166,
        3827,
        3754,
        194,
        292,
        350,
        381,
        3453,
        489,
        501,
        3270,
        511,
        533,
        3127,
        602,
        625,
        642,
        3013,
        3018,
        2991,
        2983,
        2968,
        2954,
        781,
        716,
        838,
        880,
        2818,
        2863,
        907,
        914,
        2808,
        2650,
        2630,
        2582,
        2500,
        2378,
        2337,
        2317,
        2169,
        2138,
        2126,
        2117,
        2028,
        2003,
        1934,
        1914,
        1903,
        1825,
        1810,
        1720,
        1616,
        1612,
        1339,
        1316,
    ],
    "label",
] = "!"

# 716...
# we use what GPT leans towards 2292 ,1825

df_openai_gpt_4_turbo.loc[1229]["response"]
df_openai_gpt_4_turbo.query("label == '?'")

# %%
# Map back '!' flag to '?'
df_openai_gpt_4_turbo["label"] = df_openai_gpt_4_turbo["label"].replace(
    "!",
    "?",
)

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
df_merge_all_runs = df_merge_all_runs.merge(
    df_openai_gpt_4_turbo[df_openai_gpt_4_turbo["gpt_4_label"] != "?"][
        ["text", "gpt_4_label"]
    ],
    on="text",
)
df_merge_all_runs_with_errors = df_merge_all_runs_with_errors.merge(
    df_openai_gpt_4_turbo[["text", "gpt_4_label"]],
    on="text",
)


ground_truth = df_merge["label"].astype(int)
gpt_4_label = df_merge["gpt_4_label"].astype(int)

# %%
print(
    "F1-Score with GPT 4 turbo with (4 shot CoT): ",
    f1_score(ground_truth, gpt_4_label),
)
print(
    "Precision with GPT 4 turbo with (4 shot CoT): ",
    precision_score(ground_truth, gpt_4_label),
)
print(
    "Recall with GPT 4 turbo with (4 shot CoT): ",
    recall_score(ground_truth, gpt_4_label),
)
print(
    "Accuracy with GPT 4 turbo with (4 shot CoT): ",
    accuracy_score(ground_truth, gpt_4_label),
)

# %% [markdown]
# # Llama-2-7b-chat-hf

# %%
df_Llama_2_7b.query("label == '?'")


# %%
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
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_Llama_2_7b["label"] = df_Llama_2_7b.apply(update_label, axis=1)
df_Llama_2_7b.query("label == '?'")

# %%
df_Llama_2_7b.loc[
    [3948, 3258, 2249, 2115, 1684, 1338, 864, 829, 410, 379],
    "label",
] = "BIASED"
df_Llama_2_7b.loc[
    [
        3917,
        3849,
        3797,
        3795,
        3746,
        3676,
        3664,
        3539,
        3373,
        3350,
        3096,
        3089,
        3037,
        2873,
        2658,
        2504,
        2227,
        2101,
        1861,
        2746,
        1556,
        1510,
        1455,
        1449,
        1290,
        1083,
        982,
        700,
        659,
        654,
        317,
        200,
        114,
        760,
    ],
    "label",
] = "NOT BIASED"


# df_Llama_2_7b.loc[760]['response']
df_Llama_2_7b.query("label == '?'")

# %%
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

# %%
print(
    "F1-Score with llama 7b (4 shot CoT): ",
    f1_score(ground_truth, llama_7b_label),
)
print(
    "Precision with llama 7b (4 shot CoT): ",
    precision_score(ground_truth, llama_7b_label),
)
print(
    "Recall with llama 7b (4 shot CoT): ",
    recall_score(ground_truth, llama_7b_label),
)
print(
    "Accuracy with llama 7b (4 shot CoT): ",
    accuracy_score(ground_truth, llama_7b_label),
)

# %% [markdown]
# # Llama-2-13b-chat-hf

# %%
df_Llama_2_13b.query("label == '?'")


# %%
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
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_Llama_2_13b["label"] = df_Llama_2_13b.apply(update_label, axis=1)
df_Llama_2_13b.query("label == '?'")

# %%
df_Llama_2_13b.loc[
    [
        3946,
        3870,
        3750,
        3733,
        3650,
        2189,
        2119,
        2111,
        1789,
        1432,
        1111,
        755,
        420,
        236,
    ],
    "label",
] = "BIASED"
df_Llama_2_13b.loc[
    [
        3981,
        3904,
        3863,
        3719,
        3539,
        3531,
        3424,
        3374,
        3159,
        2908,
        2883,
        2875,
        2551,
        2533,
        2179,
        1757,
        1749,
        1563,
        1385,
        1230,
        1133,
        1004,
        948,
        640,
        478,
        454,
        310,
        262,
        256,
        195,
        116,
        46,
    ],
    "label",
] = "NOT BIASED"

# undefined
df_Llama_2_13b.loc[46]["response"]
df_Llama_2_13b.query("label == '?'")

# %%
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

# %%
print(
    "F1-Score with TODO with (4 shot CoT): ",
    f1_score(ground_truth, llama_13b_label),
)
print(
    "Precision with TODO with (4 shot CoT): ",
    precision_score(ground_truth, llama_13b_label),
)
print(
    "Recall with TODO with (4 shot CoT): ",
    recall_score(ground_truth, llama_13b_label),
)
print(
    "Accuracy with TODO with (4 shot CoT): ",
    accuracy_score(ground_truth, llama_13b_label),
)

# %% [markdown]
# # Mistral-7B-Instruct-v0.1

# %%
df_mistral_7b.query("label == '?'")


# %%
# preprocessing
def update_label(row):
    if (
        row["response"].startswith("The sentence is biased")
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
            "The sentence above is classified as biased",
        )
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is labeled as biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("The sentence is classified as biased")
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        row["response"].startswith("This sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is NOT BIASED")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith(
            "The sentence is considered not biased",
        )
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence is not biased")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence presents factual")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        row["response"].startswith("The sentence presents a factual statement")
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_mistral_7b["label"] = df_mistral_7b.apply(update_label, axis=1)
df_mistral_7b.query("label == '?'")

# %%
df_mistral_7b.loc[[13, 43, 3858, 1465], "label"] = "BIASED"
df_mistral_7b.loc[
    [20, 26, 33, 42, 68, 78, 93, 3514, 3409, 2871, 2627, 1757, 639, 133],
    "label",
] = "NOT BIASED"

# undetermable 3771 but tends to bias
# undetermable 3722, 3473, 3366, 2770, 1710, 1506

df_mistral_7b.loc[133]["response"]
df_mistral_7b.query("label == '?'")

# %%

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

# %%
print(
    "F1-Score with Mistral-7B-Instruct-v0.1 with (2 shot CoT): ",
    f1_score(ground_truth, df_mistral_7b_label),
)
print(
    "Precision with Mistral-7B-Instruct-v0.1 with (2 shot CoT): ",
    precision_score(ground_truth, df_mistral_7b_label),
)
print(
    "Recall with Mistral-7B-Instruct-v0.1 with (2 shot CoT): ",
    recall_score(ground_truth, df_mistral_7b_label),
)
print(
    "Accuracy with Mistral-7B-Instruct-v0.1 with (2 shot CoT): ",
    accuracy_score(ground_truth, df_mistral_7b_label),
)

# %% [markdown]
# # Mixtral-8x7B

# %%
df_mixtral_8x7b.query("label == '?'")


# %%
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
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_mixtral_8x7b["label"] = df_mixtral_8x7b.apply(update_label, axis=1)
df_mixtral_8x7b.query("label == '?'")

# %%
df_mixtral_8x7b.loc[
    [
        9,
        235,
        238,
        3773,
        3819,
        452,
        3740,
        527,
        607,
        663,
        748,
        954,
        987,
        1017,
        3464,
        3523,
        3227,
        2974,
        2966,
        2925,
        2607,
        2576,
        2517,
        3313,
        2336,
        2289,
        2219,
        2165,
        2091,
        1988,
        1977,
        1734,
        1696,
        1638,
        1562,
        1274,
        1223,
        1134,
    ],
    "label",
] = "BIASED"
df_mixtral_8x7b.loc[
    [
        26,
        75,
        115,
        125,
        3967,
        3981,
        3995,
        4005,
        4011,
        156,
        179,
        216,
        3759,
        3946,
        3953,
        275,
        383,
        401,
        477,
        3694,
        3702,
        3709,
        3716,
        483,
        592,
        649,
        3617,
        3643,
        754,
        834,
        900,
        3581,
        3597,
        3662,
        3664,
        3674,
        1057,
        1094,
        3316,
        3400,
        3420,
        3258,
        3231,
        3229,
        3076,
        3022,
        2922,
        2869,
        2748,
        2653,
        2573,
        2504,
        2337,
        2319,
        2270,
        2234,
        2202,
        2199,
        2102,
        2017,
        1968,
        1884,
        1764,
        1660,
        1641,
        1630,
        1612,
        1534,
        1493,
        1474,
        1385,
        1370,
        1301,
        1256,
        1206,
        1133,
        1095,
    ],
    "label",
] = "NOT BIASED"

# undefined 2964, 2630

df_mixtral_8x7b.loc[1095]["response"]
df_mixtral_8x7b.query("label == '?'")

# %%
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

# %%
print(
    "F1-Score with mixtral_8x7b with (4 shot CoT): ",
    f1_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Precision with mixtral_8x7b with (4 shot CoT): ",
    precision_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Recall with mixtral_8x7b with (4 shot CoT): ",
    recall_score(ground_truth, df_mixtral_8x7b_label),
)
print(
    "Accuracy with mixtral_8x7b with (4 shot CoT): ",
    accuracy_score(ground_truth, df_mixtral_8x7b_label),
)

# %% [markdown]
# # OpenChat_3.5

# %%
df_openchat_3_5.query("label == '?'")


# %%
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
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_openchat_3_5["label"] = df_openchat_3_5.apply(update_label, axis=1)
df_openchat_3_5.query("label == '?'")

# %%
df_openchat_3_5.loc[[1697], "label"] = "BIASED"
df_openchat_3_5.loc[[989], "label"] = "NOT BIASED"

# undefined 2074

df_openchat_3_5.loc[2074]["response"]


# %%
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

# %%
print(
    "F1-Score with OpenChat 3.5 with (4 shot CoT): ",
    f1_score(ground_truth, openchat_label),
)
print(
    "Precision with OpenChat 3.5 with (4 shot CoT): ",
    precision_score(ground_truth, openchat_label),
)
print(
    "Recall with OpenChat 3.5 with (4 shot CoT): ",
    recall_score(ground_truth, openchat_label),
)
print(
    "Accuracy with OpenChat 3.5 with (4 shot CoT): ",
    accuracy_score(ground_truth, openchat_label),
)

# %% [markdown]
# # zephyr-7b-beta

# %%
df_zephyr_7b_beta.query("label == '?'")


# %%
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
    elif (
        find_first_occurrence(row["response"]) == "The answer is BIASED"
        and row["label"] == "?"
    ):
        return "BIASED"
    elif (
        find_first_occurrence(row["response"]) == "The answer is NOT BIASED"
        and row["label"] == "?"
    ):
        return "NOT BIASED"
    else:
        return row["label"]


df_zephyr_7b_beta["label"] = df_zephyr_7b_beta.apply(update_label, axis=1)
df_zephyr_7b_beta.query("label == '?'")

# %%
df_zephyr_7b_beta.loc[
    [
        9,
        156,
        3626,
        3663,
        3750,
        420,
        3606,
        3916,
        3322,
        3313,
        2853,
        2819,
        2812,
        2674,
        2299,
        2268,
        1753,
        1306,
        1078,
        1046,
        927,
        732,
        723,
        665,
        452,
    ],
    "label",
] = "BIASED"
df_zephyr_7b_beta.loc[
    [
        33,
        42,
        48,
        3932,
        238,
        393,
        399,
        406,
        3538,
        3602,
        3932,
        3471,
        3460,
        3332,
        3316,
        3268,
        3192,
        3179,
        2866,
        2796,
        2746,
        2705,
        2459,
        2318,
        2199,
        2165,
        2145,
        2025,
        2016,
        1981,
        1933,
        1873,
        1860,
        1661,
        1575,
        1541,
        1449,
        1320,
        1258,
        1096,
        1046,
        991,
        935,
        888,
        689,
        659,
        576,
        470,
    ],
    "label",
] = "NOT BIASED"

# undecideable 1465
# MIXED-> BIASED 2299

df_zephyr_7b_beta.loc[1465]["response"]
df_zephyr_7b_beta.query("label == '?'")

# %%
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

# %%
print(
    "F1-Score with zephyr beta (4 shot): ",
    f1_score(ground_truth, zephyr_label),
)
print(
    "Precision with zephyr beta (4 shot): ",
    precision_score(ground_truth, zephyr_label),
)
print(
    "Recall with zephyr beta (4 shot): ",
    recall_score(ground_truth, zephyr_label),
)
print(
    "Accuracy with zephyr beta (4 shot): ",
    accuracy_score(ground_truth, zephyr_label),
)

# %%
# safe the file
df_merge_all_runs_with_errors.to_csv("./all_runs_with_errors.csv", index=False)
