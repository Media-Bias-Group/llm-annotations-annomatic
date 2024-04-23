import logging
import uuid

from src.utils_ import unify_site_name, to_parquet

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RAW_PATH = "data/raw"
INPUT_PATH = "data/extract/tmp"
OUTPUT_PATH = "data/extract/output"


def _community_trust(fdbck):
    agrees, disagrees = fdbck.split("/")
    return int(agrees) / (int(agrees) + int(disagrees))


def _community_volume(fdbck):
    agrees, disagrees = fdbck.split("/")
    return int(agrees) + int(disagrees)


@to_parquet(f"{OUTPUT_PATH}/outlets.parquet")
def main():
    d1 = pd.read_parquet(f"{INPUT_PATH}/allsides_snapshot.parquet")
    d2 = pd.read_parquet(f"{RAW_PATH}/adfontes_snapshot.parquet")

    d1["uni_source"] = d1["news_source"].apply(unify_site_name)
    d2["uni_source"] = d2["source"].apply(unify_site_name)

    merged = d1.merge(d2, on="uni_source")
    merged = merged[~merged["news_link"].isna()]
    merged["community_trust"] = merged["community_feedback"].apply(
        _community_trust,
    )
    merged["community_feedback"] = merged["community_feedback"].apply(
        _community_volume,
    )
    scaler = MinMaxScaler()
    merged["reliability"] = scaler.fit_transform(merged[["reliability"]])

    merged = merged[merged.bias_rating != "Mixed"]

    neutral = merged[merged.bias_rating == "Center"].sort_values(
        "community_feedback",
        ascending=False,
    )[:30]
    biased = merged[merged.bias_rating != "Center"]

    merged = pd.concat([biased, neutral])

    merged["outlet_id"] = [str(uuid.uuid4()) for _ in range(len(merged))]
    return merged


if __name__ == "__main__":
    main()
