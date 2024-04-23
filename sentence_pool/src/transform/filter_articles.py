import pandas as pd
from langdetect import detect
from src.utils_ import to_parquet
from tqdm import tqdm

INPUT_PATH = "data/extract/output"
OUTPUT_PATH = "data/transform/tmp"
tqdm.pandas()


@to_parquet(f"{OUTPUT_PATH}/articles.parquet")
def main():
    df = pd.read_parquet(f"{INPUT_PATH}/articles.parquet")
    df = df[df.text != ""]
    df = df[~df.title.isna()]
    df = df[~df[["text", "title", "outlet_id"]].duplicated()]
    df["approx_len"] = df["text"].apply(lambda x: len(x.split(".")))
    df = df[df.approx_len > 2]
    lang_mask = df["text"].progress_apply(lambda x: detect(x) == "en")
    df = df[lang_mask]
    return df


if __name__ == "__main__":
    main()
