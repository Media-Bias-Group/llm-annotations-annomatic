import pandas as pd
import seaborn as sns
from datasets import load_dataset
from fuzzywuzzy import fuzz
import huggingface_hub


def fuzzify(text):
    """
    A workaround when IDs are missing. Applies fuzzy string matching to the given text.
    """
    return fuzz._process_and_sort(text, force_ascii=True)


# load BABE with full
babe = load_dataset("mediabiasgroup/BABE", revision="v1")["train"].to_pandas()

# load new annotations from three annotators
d1 = pd.read_csv("mismatched-reannotation/expert1.csv")[:-1]
d2 = pd.read_csv("mismatched-reannotation/expert2.csv")
d3 = pd.read_csv("mismatched-reannotation/expert3.csv")[:-1]

reann = pd.DataFrame(
    {
        "text": d1.text,
        "expert1": d1.expert1,
        "expert2": d2.expert2,
        "expert3": d3.expert3,
        "babe": d1.label,
    },
)

reann["combined"] = reann[["expert1", "expert2", "expert3"]].mode(axis=1)[0]

babe["fuzz"] = babe.text.apply(fuzzify)
reann["fuzz"] = reann.text.apply(fuzzify)

# match on fuzzified string as IDs are not available in reannotations
merged = babe.merge(reann, how="left", on="fuzz")

# update babe with new annotations
merged["label"].update(merged["combined"])
merged.drop_duplicates("text_x")

babe = merged[
    [
        "text_x",
        "news_link",
        "outlet",
        "topic",
        "type",
        "label",
        "label_opinion",
        "biased_words",
        "fuzz",
    ]
].rename(columns={"text_x": "text"})


# push to huggingface
# babe.push_to_hub(
#     "anonymous",
#     commit_message="Upload reannotation version BABE v2",
#     commit_description="This version is the same as version v1 in size but includes 400 reannotated sentences from Expert1,2 and 3.",
# )
# huggingface_hub.create_tag("anonymous/BABE", tag="v2", repo_type="dataset")
