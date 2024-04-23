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
d1 = pd.read_csv("mismatched-reannotation/timo_.csv")[:-1]
d2 = pd.read_csv("mismatched-reannotation/christoph_.csv")
d3 = pd.read_csv("mismatched-reannotation/tomas_.csv")[:-1]

reann = pd.DataFrame(
    {
        "text": d1.text,
        "timo": d1.Timo,
        "christoph": d2.Christoph,
        "tomas": d3.Tomas,
        "babe": d1.label,
    },
)

reann["combined"] = reann[["timo", "tomas", "christoph"]].mode(axis=1)[0]

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
#     "_mediabiasgroup/BABE",
#     commit_message="Upload reannotation version BABE v2",
#     commit_description="This version is the same as version v1 in size but includes 400 reannotated sentences from Tomas,Timo and Christoph.",
# )
# huggingface_hub.create_tag("mediabiasgroup/BABE", tag="v2", repo_type="dataset")
