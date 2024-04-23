import logging
import os
import random
import subprocess
import time
import uuid

import newspaper
import pandas as pd
from newspaper import ArticleException
from tqdm import tqdm
from src.utils_ import to_parquet
logging.getLogger("newspaper").setLevel(logging.CRITICAL)

OUTPUT_PATH = "data/extract/output"
INPUT_PATH = "data/extract/output"
TMP_PATH = "data/extract/tmp"


def _scrape_source_articles(outlet_link):
    try:
        source = newspaper.build(outlet_link, memoize_articles=False)
    except:
        # logging.ERROR(f"Can't build {outlet_link}. Not an URL.")
        return None

    rowlist = []
    i = 0

    for article in tqdm(source.articles):
        try:
            article.download()
            article.parse()
            rowlist.append(
                {
                    "text": article.text,
                    "title": article.title,
                },
            )
            if i == 1000:
                break
            i += 1
        except:
            continue

    return pd.DataFrame(rowlist)



@to_parquet(f"{OUTPUT_PATH}/articles.parquet")
def main():
    outlets_df = pd.read_parquet(f"{INPUT_PATH}/outlets.parquet").sample(
        frac=1
    )

    # scrape articles
    for _, outlet_row in tqdm(outlets_df.iterrows(), total=len(outlets_df)):
        if os.path.isfile(
            f"{TMP_PATH}/articles_{outlet_row['uni_source']}.parquet"
        ):
            print("Already scraped")
            continue
        articles_df = _scrape_source_articles(
            outlet_link=outlet_row["news_link"],
        )
        if articles_df is None:
            continue
        articles_df["outlet_id"] = [outlet_row["outlet_id"]] * len(articles_df)
        articles_df["article_id"] = [
            str(uuid.uuid4()) for _ in range(len(articles_df))
        ]
        articles_df.to_parquet(
            f"{TMP_PATH}/articles_{outlet_row['uni_source']}.parquet",
        )

    # merge articles
    article_dataframes = []
    for data in os.listdir(f"{TMP_PATH}"):
        if data.split("_")[0] == "articles":
            article_dataframes.append(
                pd.read_parquet(f"{TMP_PATH}/{data}")
            )

    articles_df = pd.concat(article_dataframes)
    subprocess.run(["rm", "-r", f"{TMP_PATH}"])
    return articles_df

if __name__ == "__main__":
    main()
